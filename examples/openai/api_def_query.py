"""
自动化测试接口实体
http 请求方法枚举类
@author chaoxin.lu
@version V 1.0
"""
# -*- coding: UTF-8 -*-

import json
import os
import re
import shutil
import time
import uuid
from threading import Lock

import cache3
import jinja2
import pymysql
import pytest
import requests
import yaml
from oauth2 import OAuth2Client
from pymysql import Connection

from entity.types import ApiTestCase, Res
from handler.logger_handler import logger
from s3.s3_svc import S3Storage


def get_data(json_resp, key: str = None):
    if json_resp.status_code == 200:
        json_data = json_resp.json()
        if json_data.get('code', None) == '0':
            data = json_data.get('data', None)
            if key and isinstance(data, dict):
                return data.get(key, None)
            else:
                return data
    return None


def validate(doc: dict) -> bool:
    return doc.get('isFolder') == 0 and doc.get('isDeleted') == 0 and doc.get('isShow') == 1


class ApiDefQuery:
    def __init__(self, connection: Connection = None):
        super().__init__()
        self.retries = 3
        self.torna_endpoint = os.getenv('TORNA_ENDPOINT', 'https://papi.piston.ink')
        self.torna_user = os.getenv('TORNA_USER', 'lucx')
        self.torna_pwd = os.getenv('TORNA_PWD', 'f5b941c45f213a735cd55884468332b9')
        if connection:
            self._connection = connection
        else:
            self._connection = pymysql.connect(
                host=os.getenv('MYSQL_HOST', 'gzv-dev-maria-1.piston.ink'),
                port=int(os.getenv('MYSQL_PORT', 3306)),
                user=os.getenv('MYSQL_USER', 'pistonint_cloud'),
                password=os.getenv('MYSQL_PWD', 'p!s@t$o$n.i.n.t#@!mysql'),
                database=os.getenv('MYSQL_DB', 'pistonint_automated_test'),
                charset='utf8')
        self._oauth2_user = os.getenv('OAUTH2_USER')
        self._oauth2_pwd = os.getenv('OAUTH2_PWD')
        self._oauth2_authz = os.getenv('OAUTH2_AUTHORIZATION')
        self._cache_lock = Lock()
        self._cache = cache3.Cache(name='torna_token')

    def get_api_list(self) -> list[ApiTestCase]:
        r: Res[list[ApiTestCase]] = self.get_api_info
        if r.code == 1:
            return r.data
        return []

    @property
    def get_api_info(self) -> Res[list[ApiTestCase]]:
        sql = """
            SELECT `id`,`batch_id`,`module_id`,`module_name`,`execute_env`,`oauth2_env`,`project_id`,`project_name`  
                  FROM automated_test_job_def 
                  WHERE `status` <> 2 
                  ORDER BY `create_time` DESC 
                  limit 1
                  """
        with self._connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchone()
            if not result:
                return Res(0, '无任务')
            pk = result[0]
            batch_id = result[1]
            module_id = result[2]
            module_name = result[3]
            execute_env = result[4]
            oauth2_env = result[5]
            project_id = result[6]
            project_name = result[7]
            update_sql = "update automated_test_job_def set `status` = 1 where id = %d" % pk
            try:
                cursor.execute(update_sql)
                self._connection.commit()
            except Exception as e:
                self._connection.rollback()
                logger().error(e)
                return Res(0, str(e))
        r = self.get_apis(project_id, project_name, module_id, module_name, execute_env, batch_id, oauth2_env)
        if r.code != 1:
            logger().error(f"查询不到接口信息，module_id：{module_id}，env:{execute_env}, error: {r.msg}")
        return r

    def get_torna_access_token(self) -> Res[str]:
        cache_key = 'torna_access_token'
        token = self._cache.get(cache_key, None)
        if not token and self._cache_lock.acquire(blocking=True, timeout=10000):
            token = self._cache.get(cache_key, None)
            if not token:
                login_resp = requests.post(
                    url=f"{self.torna_endpoint}/system/login",
                    json={'username': self.torna_user, 'password': self.torna_pwd, 'source': 'register'},
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer'},
                    verify=False)
                token = get_data(login_resp, 'token')
                if not token:
                    logger().error(
                        f"自动化测试：登录失败，endpoint:{self.torna_endpoint},用户名:{self.torna_user},error:{login_resp.json().get('msg', '')}")
                    return Res(0, '登录失败')
                access_token = 'Bearer {}'.format(token)
                self._cache.ex_set(cache_key, access_token, 60)
                self._cache_lock.release()
        return Res(0, token)

    def get_piston_access_token(self, oauth2_env) -> Res[str]:
        try:
            sql = "select url from automated_test_oauth2_inf where `env` = '%s'" % oauth2_env
            cursor = self._connection.cursor()
            cursor.execute(sql)
            res = cursor.fetchone()
            oauth2_url = None
            for i in res:
                oauth2_url = i
            oauth2_client = OAuth2Client(oauth2_url)
            resp = oauth2_client.get_access_token_header(self._oauth2_user, self._oauth2_pwd, self._oauth2_authz)
            if resp is not None:
                return Res(1, resp)
            return Res(0, '获取token失败')
        except Exception as e:
            logger().error("获取token失败", e)
            return Res(0, '获取token失败')

    def get_first_module(self, project_id) -> dict:
        r = self.get_torna_access_token()
        if r.code != 1:
            return {}
        resp = requests.get(url=f"{self.torna_endpoint}/module/list",
                            params={'projectId': project_id},
                            headers={'Authorization': r.data},
                            verify=False)
        module_list = get_data(resp)
        if isinstance(module_list, list):
            return module_list[0]
        return {}

    def get_project_info(self, project_id) -> dict:
        r = self.get_torna_access_token()
        if r.code != 1:
            return {}
        resp = requests.get(url=f"{self.torna_endpoint}/project/info",
                            params={'projectId': project_id},
                            headers={'Authorization': r.data},
                            verify=False)
        return get_data(resp)

    def get_module_info(self, module_id) -> dict:
        r = self.get_torna_access_token()
        if r.code != 1:
            return {}
        resp = requests.get(url=f"{self.torna_endpoint}/module/info",
                            params={'moduleId': module_id},
                            headers={'Authorization': r.data},
                            verify=False)
        return get_data(resp)

    def convert_doc(self, doc_details, test_url, project_name, module_name, batch_id, oauth2_env) -> list[ApiTestCase]:
        test_doc_inf_list: list[ApiTestCase] = []
        for doc_detail in doc_details:
            doc_id = doc_detail.get('id')
            content_type = doc_detail.get('contentType')
            method = doc_detail.get('httpMethod')
            doc_url: str = '{}{}'.format(test_url, doc_detail.get('url'))
            req_headers = {}
            req_params = {}
            for global_header in doc_detail.get('globalHeaders'):
                req_headers[global_header.get('name')] = global_header.get('example')
            for global_param in doc_detail.get('globalParams'):
                req_params[global_param.get('name')] = global_param.get('example')
            for query_param in doc_detail.get('queryParams'):
                req_params[query_param.get('name')] = query_param.get('example')

            temp_req_params = []
            for request_param in doc_detail.get('headerParams'):
                if request_param.get('style') == 1:
                    req_headers[request_param.get('name')] = request_param.get('example')

            for request_param in doc_detail.get('pathParams'):
                if request_param.get('style') == 0:
                    path_param_name = request_param.get('name')
                    path_param_value = request_param.get('example')
                    pattern = re.compile(r'({' + path_param_name + '.*})')
                    doc_url = re.sub(pattern=pattern, repl=path_param_value, string=doc_url)
            for request_param in doc_detail.get('requestParams'):
                # style类型
                # 0：path, 1：header， 2：请求参数，3：返回参数，4：错误码
                if request_param.get('style') == 2:
                    temp_req_params.append(request_param)

            result = self.convert_param(temp_req_params)
            for k, v in result.items():
                req_params[k] = v
            req_headers['Authorization'] = self.get_piston_access_token(oauth2_env)
            doc_name = doc_detail.get('name')
            name = '项目名称：{}\n模块名称：{}\n接口名称：{}'.format(project_name, module_name, doc_detail.get('name'))
            test_doc_inf_list.append(ApiTestCase(
                batch_id=batch_id,
                doc_id=doc_id,
                doc_name=doc_name,
                name=name,
                project_name=project_name,
                content_type=content_type,
                method=method,
                url=doc_url,
                req_headers=req_headers,
                req_params=req_params
            ))
        return test_doc_inf_list

    def get_apis(self, project_id: str, project_name: str, module_id: str, module_name: str, req_env: str,
                 batch_id: str, oauth2_env: str) -> Res[list[ApiTestCase]]:
        r = self.get_torna_access_token()
        if r.code != 1:
            return Res(0, r.msg)
        access_token = r.data
        env_resp = requests.get(url=f"{self.torna_endpoint}/module/environment/list",
                                params={'moduleId': module_id},
                                headers={'Authorization': access_token},
                                verify=False)
        test_url = None
        envs = get_data(env_resp)
        for env in envs:
            if env.get('name', None) == req_env:
                test_url = env.get('url', None)
                break
        if not test_url:
            return Res(0, f"自动化测试：查询测试环境失败，env:{req_env}")
        doc_list_resp = requests.get(url=f"{self.torna_endpoint}/doc/list",
                                     params={'moduleId': module_id},
                                     headers={'Authorization': access_token},
                                     verify=False)
        doc_list = get_data(doc_list_resp)
        if not doc_list:
            return Res(0, f"自动化测试：查询接口列表失败，moduleId:{module_id}")
        ids = list(map(lambda t: t.get('id'), filter(lambda x: validate(x), doc_list)))
        doc_details_resp = requests.post(url=f"{self.torna_endpoint}/doc/detail/search",
                                         json={'docIdList': ids},
                                         headers={'Authorization': access_token, 'Content-Type': 'application/json'},
                                         verify=False)
        doc_details = get_data(doc_details_resp)
        if not doc_details:
            return Res(0, f"接口为空，project_id:{project_id},moduleId:{module_id}")
        apis: list[ApiTestCase] = self.convert_doc(doc_details, test_url, project_name, module_name, batch_id,
                                                   oauth2_env)
        return Res(1, '查询成功', apis)

    def convert_param(self, orig_params: list):
        group_by_parent = {}
        for param in orig_params:
            parent_id = param.get('parentId')
            temp_list = group_by_parent.get(parent_id, None)
            if temp_list:
                temp_list.append(param)
            else:
                group_by_parent[parent_id] = [param]
        current = {}
        for param in orig_params:
            child_dir = self.find_child(param.get('id'), group_by_parent)
            te = param.get('type').lower()
            label = te.find('array') != -1 or te.find('list') != -1
            if not child_dir:
                if label:
                    current[param.get('name')] = param.get('example').split(',')
                else:
                    current[param.get('name')] = param.get('example')
            else:
                if label:
                    current[param.get('name')] = [child_dir]
                else:
                    current[param.get('name')] = child_dir
        return current

    def find_child(self, curr_id: str, cache: dict):
        children = cache.get(curr_id, {})
        curr_dir = {}
        if not children:
            return curr_dir
        for child in children:
            temp = self.find_child(child['id'], cache)
            te = child.get('type').lower()
            label = te.find('array') != -1 or te.find('list') != -1
            if not temp:
                if label:
                    curr_dir[child['name']] = child['example'].split(',')
                else:
                    curr_dir[child['name']] = child['example']
            else:
                curr_dir[child['name']] = temp
        return curr_dir


class AutoTestSvc:

    def __init__(self):
        self._oauth_cli = OAuth2Client()
        self._oauth2_user = os.getenv('OAUTH2_USER')
        self._oauth2_pwd = os.getenv('OAUTH2_PWD')
        self._oauth2_authz = os.getenv('OAUTH2_AUTHORIZATION')
        self._connection = pymysql.connect(
            host=os.getenv('MYSQL_HOST', 'gzv-dev-maria-1.piston.ink'),
            port=int(os.getenv('MYSQL_PORT', 3306)),
            user=os.getenv('MYSQL_USER', 'pistonint_cloud'),
            password=os.getenv('MYSQL_PWD', 'p!s@t$o$n.i.n.t#@!mysql'),
            database=os.getenv('MYSQL_DB', 'pistonint_automated_test'),
            charset='utf8')
        self._doc_inf_query = ApiDefQuery(self._connection)
        self._dingtalk_hook_endpoint = os.getenv("DINGTALK_HOOK_SVC_ENDPOINT",
                                                 "https://dingtalk-hook.gzv-k8s.piston.ink")
        work_dir = os.getenv('WORK_DIR',
                             os.path.dirname(os.path.abspath(os.path.split(os.path.realpath(__file__))[0])))
        self._base_report_path = work_dir
        self._base_html_dir = os.path.join(work_dir, 'html')
        self._s3_storage = S3Storage()
        parent_path = os.path.abspath('.')
        loader = jinja2.FileSystemLoader(searchpath=parent_path + "/templates")
        self._template_env = jinja2.Environment(loader=loader)
        self._k8s_proxy_endpoint = os.getenv("K8S_DEPLOY_PROXY_ENDPOINT",
                                             "https://k8s-rest-proxy.gzv-k8s.piston.ink")

    def get_connection(self):
        return self._connection

    def close(self):
        self._connection.close()

    def add_job(self, req: dict) -> Res:
        batch_id = None
        try:
            log_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
            batch_id = '{}-{}'.format(log_time, uuid.uuid4())
            module_id = req.get('module_id', None)
            execute_env = req.get('execute_env', None)
            oauth2_env = req.get('oauth2_env', None)

            if not module_id or not execute_env or not oauth2_env:
                return Res(0, '必须指定execute_env、oauth2_env、module_id')
            module_info = self._doc_inf_query.get_module_info(module_id)
            if not module_info:
                return Res(0, '找不到module, module id：{}'.format(module_id))
            module_name = module_info.get('name', None)
            project_id = module_info.get('projectId', None)
            project_info = self._doc_inf_query.get_project_info(project_id)
            if not project_info:
                return Res(0, '找不到project, project id：{}, module id：{}'.format(project_id, module_id))
            project_name = project_info.get('name', None)
            sql = """
            insert into automated_test_job_def(
                                batch_id, project_id, project_name, module_id, module_name, execute_env, oauth2_env)
                                values (%s, %s, %s, %s, %s, %s, %s)
                                """
            with self.get_connection().cursor() as cursor:
                cursor.execute(sql,
                               (batch_id, project_id, project_name, module_id, module_name, execute_env, oauth2_env,))
                self.get_connection().commit()
            return Res(1, '新增自动化测试任务成功',
                       {'batch_id': batch_id, 'project_id': project_id, 'project_name': project_name})

        except Exception as e:
            self.get_connection().rollback()
            with self.get_connection().cursor() as cursor:
                update_sql = """
                            update automated_test_job_def 
                            set `status` = 3
                            where batch_id = '%s'""" % batch_id
                cursor.execute(update_sql)
                self.get_connection().commit()
            logger().error('新增自动化测试任务失败', e)
        return Res(0, '新增自动化测试任务失败')

    def invoke(self, project_name: str, bid: str, mid: str, execute_env: str, oauth2_env: str):
        base_dir = os.path.join(self._base_report_path, 'report')
        zip_path = os.path.join(self._base_report_path, 'report', 'zip')
        pytest_result = os.path.join(base_dir, 'json_result', bid, project_name)
        allure_result = os.path.join(base_dir, 'allure', 'html', bid, project_name)
        html_dir = os.path.join(self._base_html_dir, project_name)
        os.system('mkdir -p {} {} {}'.format(pytest_result, allure_result, zip_path))
        os.system('rm -rf {}/* {}/* {}'.format(pytest_result, allure_result, html_dir))
        logger().info('batch_id:{},module_id:{},execute_env:{},oauth2_env:{}'.format(bid, mid, execute_env, oauth2_env))

        allure_command = 'allure generate {} -o {} --clean'.format(pytest_result, allure_result)
        # 执行测试
        pytest.main([
            '-vs', 'scripts/api_auto_test.py',
            '--alluredir', pytest_result,
            '--allure_command', allure_command,
            '--batch_id', bid,
            '--module_id', mid,
            '--execute_env', execute_env,
            '--oauth2_env', oauth2_env
        ])
        # 生成报告
        os.system(allure_command)

        zipfile = zip_path + '/' + bid
        shutil.make_archive(base_name=zipfile, format='zip', root_dir=allure_result)
        now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
        shutil.copytree(allure_result, html_dir)
        html_prefix = project_name + "/" + bid
        self._s3_storage.upload_files(html_dir, html_prefix)
        name = project_name + "-" + bid
        if self.deploy_nginx(project_name, name, html_prefix):
            # {{name}}.gzv-k8s.piston.ink
            allure_report_url = "https://{}.gzv-k8s.piston.ink".format(name)
            text = "- 服务名称：{}\n- [测试报告详情]({})".format(project_name, allure_report_url)
            self.alert(text)
        with self.get_connection().cursor() as cursor:
            update_sql = """
                update automated_test_job_def 
                                set `status` = 2, update_time = '%s' 
                                where batch_id = '%s'
                                """ % (now, bid)
            cursor.execute(update_sql)
            self.get_connection().commit()
        self.delete_history_resource(project_name)

    def alert(self, text: str):
        dingtalk_hook_url = self._dingtalk_hook_endpoint + "/message/markdown"
        req_body = {
            "secret": os.getenv("DINGTALK_SECRET",
                                "SECe4004e55fb9c3442c1e94245e96cd57be712da52ca1d9c93d0a2f0f30471bfa5"),
            "token": os.getenv("DINGTALK_TOKEN",
                               "4a48d9ee084e205940ff7b6346d8eb57e05068ac86ba6fcd9172ae305a4ca02b"),
            "msgtype": 2,
            "markdownMsg": {
                "markdown": {
                    "title": "自动化测试报告",
                    "text": text
                },
                "at": {"isAtAll": True}
            }
        }

        req_headers = {"Content-Type": "application/json"}
        requests.post(url=dingtalk_hook_url,
                      json=req_body,
                      headers=req_headers,
                      verify=False)

    def deploy_nginx(self, project_name: str, name: str, html_prefix: str) -> bool:
        m = {
            "download-cm.yaml": "config-map",
            "deployment.yaml": "deploy",
            "ingress.yaml": "ingress",
            "svc.yaml": "service"
        }
        for (key, value) in m.items():
            b = self.deploy_resource(project_name, name, key, value, html_prefix)
            if not b:
                return b
        return True

    def deploy_resource(self, project_name: str, name: str, tpml: str, uri: str, html_prefix: str) -> bool:
        template = self._template_env.get_template(tpml)
        tmpl = template.render(name=name,
                               html_prefix=html_prefix,
                               project_name=project_name)
        yaml_object = yaml.safe_load(tmpl)
        data = json.dumps(yaml_object)
        token = self._oauth_cli.get_access_token_header(self._oauth2_user, self._oauth2_pwd, self._oauth2_authz)
        req_headers = {
            "Authorization": token,
            "Content-Type": "application/json"
        }
        resp = requests.post(url=self._k8s_proxy_endpoint + "/" + uri,
                             data=data,
                             headers=req_headers,
                             verify=False)
        if resp.status_code != 200:
            res = resp.json()
            logger().error(res.get('msg'))
        return resp.status_code == 200

    def delete_history_resource(self, project_name):
        token = self._oauth_cli.get_access_token_header(self._oauth2_user, self._oauth2_pwd, self._oauth2_authz)
        req_headers = {
            "Authorization": token,
            "Content-Type": "application/json"
        }
        query = """
            select project_name, batch_id from automated_test_job_def
                        where project_name = '%s'
                        order by create_time desc
                        limit 3,10
            """ % project_name
        with (self.get_connection().cursor() as cursor):
            cursor.execute(query)
            results = cursor.fetchall()
            resources = ["config-map", "deploy", "ingress", "service"]
            for result in results:
                for resource in resources:
                    try:
                        project_name = result[0]
                        batch_id = result[1]
                        url = self._k8s_proxy_endpoint + "/" + resource \
                              + "?namespace=piston-automated-test" \
                              + "&name=" + project_name + "-" + batch_id
                        resp = requests.delete(url=url,
                                               headers=req_headers,
                                               verify=False)
                        if resp.status_code == 200:
                            logger().info("delete resource[{}] succeed".format(resource))
                        else:
                            logger().error("delete failed {}".format(resp.json().get('msg', '')))
                    except Exception as e:
                        logger().error("delete failed", e)
