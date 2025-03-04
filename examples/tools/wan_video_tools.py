import os
from http import HTTPStatus

import requests
# dashscope sdk >= 1.22.1
from dashscope import VideoSynthesis

url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"


def sample_sync_call_t2v_v1(story: str):
    # call sync api, will return the result
    print('please wait...')
    rsp = VideoSynthesis.async_call(model='wanx2.1-t2v-turbo',
                                    prompt=story,
                                    duration=60 * 3,
                                    size='1280*720')
    print(rsp)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp.output.video_url)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))


# https://help.aliyun.com/zh/model-studio/developer-reference/text-to-video-api-reference?spm=a2c4g.11186623.help-menu-2400256.d_3_3_5_1.66b04d9fehn1q7
def sample_sync_call_t2v_v2(story: str):
    rsp = requests.post(
        url=url,
        json={
            "model": "wanx2.1-t2v-turbo",
            "input": {
                "prompt": f"{story}"
            },
            "parameters": {
                "size": "1280*720",
                "duration": 60 * 3

            }
        },
        headers={
            "Authorization": "Bearer " + f"{os.getenv('DASHSCOPE_API_KEY')}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json"
        }
    )
    print(rsp.json())


if __name__ == '__main__':
    text = """
量子共振波在真空里泛起淡紫色的涟漪，我握着相位步枪的手套正在结晶化。头盔显示器上跳动着猩红的警告：晶态同化率37%——这意味着我的右手已经变成硅基生命最喜欢的能量载体。
"指挥官，B区护盾还剩三分钟！"通讯器里传来爆破手乔伊的嘶吼。透过悬浮战车的防辐射玻璃，我看到十二公里外的晶簇巨塔正在脉动，那些六棱柱结构的表面流转着银河般的光带，每次闪烁都有新的硅基战兽从结晶矿脉中剥离。
    """
    sample_sync_call_t2v_v2(text)
