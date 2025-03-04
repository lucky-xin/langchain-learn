import os

import requests

# dashscope sdk >= 1.22.1


if __name__ == '__main__':
    task_id = "08ad72c9-b473-4476-acb3-3eb20c9de1d6"
    resp = requests.get(
        headers={
            "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
        },
        url=f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    )
    print(resp.json())
