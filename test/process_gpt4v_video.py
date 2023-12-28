# from IPython.display import display, Image, Audio
# from PIL import Image
import matplotlib.pyplot as plt
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import openai
from openai import AzureOpenAI
import os
import requests
import numpy as np

openai.api_key = "64435429e3c94ef2920c3560b202d26a" #os.environ.get('OPEN_AI_KEY')
openai.azure_endpoint = "https://ai-sh-jerryz-swc.openai.azure.com/" #os.environ.get('OPEN_AI_ENDPOINT')
openai.api_type = 'azure'
openai.api_version = '2023-12-01-preview'

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id = "gpt-4-vision-preview"

client = openai.AzureOpenAI(
    api_version="2023-12-01-preview",
    api_key=openai.api_key,
    azure_endpoint=openai.azure_endpoint
)


# Read Video
video = cv2.VideoCapture("../tempDir/The_Dandelion.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

if False:
    # plt.ion()
    fig = plt.figure()
    for img in base64Frames:
        # display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
        # time.sleep(0.025)
        # cv2.imshow("test", Image(base64.b64decode(img.encode("utf-8"))))
        # 将base64字符串解码为二进制数据
        img_bytes = base64.b64decode(img.encode("utf-8"))
        # 将二进制数据转换为numpy数组
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        # 将numpy数组解码为图像
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # 将BGR图像转换为RGB图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 使用matplotlib显示图像
        plt.imshow(frame_rgb)
        plt.pause(0.0416)  # 暂停一段时间，以便更新显示
    # 关闭图形显示
    plt.close(fig)

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. Please describe what you see.",
            *map(lambda x: {"image": x, "resize": 360}, base64Frames[0::110]),
        ],
    },
]
params = {
    "model": deployment_id,
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)