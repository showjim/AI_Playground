import requests

url = "https://api.siliconflow.cn/v1/audio/transcriptions"
key = "sk-xxxxxx"
audioFileName = "../tmp.wav"
audioFile = open(audioFileName, 'rb')
files = { "file": audioFile }
payload = { "model": "FunAudioLLM/SenseVoiceSmall" }
headers = {"Authorization": F"Bearer {key}"}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.json())