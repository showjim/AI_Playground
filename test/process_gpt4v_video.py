from IPython.display import display, Image, Audio
# from PIL import Image
import matplotlib.pyplot as plt
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import AzureOpenAI
import os
import requests
import numpy as np

# client = AzureOpenAI()
video = cv2.VideoCapture("../tempDir/test3.mkv")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

# display_handle = display(None, display_id=True)
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
    plt.pause(0.025)  # 暂停一段时间，以便更新显示
# 关闭图形显示
plt.close(fig)