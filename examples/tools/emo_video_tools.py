# 替换为你的 API 密钥
import requests

API_KEY = "<YOUR_API_KEY>"

# 请求头
headers = {
    "X-DashScope-Async": "enable",
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 请求数据（JSON 结构）
payload = {
    "model": "emo-v1",
    "input": {
        "image_url": "http://xxx/1.jpg",
        "audio_url": "http://xxx/1.wav",
        "face_bbox": [10, 20, 30, 40],
        "ext_bbox": [10, 20, 30, 40]
    },
    "parameters": {
        "style_level": "normal"
    }
}

# 发送 POST 请求
url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis/"
response = requests.post(url, headers=headers, json=payload)

# 处理响应
print(f"状态码: {response.status_code}")
print("响应内容:", response.json())


if __name__ == '__main__':
    text = """
量子共振波在真空里泛起淡紫色的涟漪，我握着相位步枪的手套正在结晶化。头盔显示器上跳动着猩红的警告：晶态同化率37%——这意味着我的右手已经变成硅基生命最喜欢的能量载体。
"指挥官，B区护盾还剩三分钟！"通讯器里传来爆破手乔伊的嘶吼。透过悬浮战车的防辐射玻璃，我看到十二公里外的晶簇巨塔正在脉动，那些六棱柱结构的表面流转着银河般的光带，每次闪烁都有新的硅基战兽从结晶矿脉中剥离。
    """
    sample_sync_call_t2v(text)
