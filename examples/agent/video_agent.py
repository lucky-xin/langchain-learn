# https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni
# https://help.aliyun.com/zh/model-studio/getting-started/models

from http import HTTPStatus
# dashscope sdk >= 1.22.1
from dashscope import VideoSynthesis

def sample_sync_call_t2v():
    # call sync api, will return the result
    print('please wait...')
    rsp = VideoSynthesis.call(model='wanx2.1-t2v-turbo',
                              prompt='一只小猫在月光下奔跑',
                              size='1280*720')
    print(rsp)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp.output.video_url)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))


if __name__ == '__main__':
    sample_sync_call_t2v()