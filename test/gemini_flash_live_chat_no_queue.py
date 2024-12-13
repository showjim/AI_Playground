import asyncio
import base64
import json
import os
import pyaudio
from websockets.asyncio.client import connect


class SimpleGeminiVoice:
    def __init__(self):
        self.api_key = "AIzaSyAsfEb9YCZxoD_u2t2xw4n8wp3MbmCpx5U" #os.environ.get("GEMINI_API_KEY")
        self.model = "gemini-2.0-flash-exp"
        self.uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.CHUNK = 512

    async def start(self):
        # Initialize websocket
        self.ws = await connect(
            self.uri, additional_headers={"Content-Type": "application/json"}
        )
        await self.ws.send(json.dumps({"setup": {"model": f"models/{self.model}"}}))
        await self.ws.recv(decode=False)
        print("Connected to Gemini, You can start talking now")
        # Start audio streaming
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.send_user_audio())
            tg.create_task(self.recv_model_audio())

    async def send_user_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=16000,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        while True:
            data = await asyncio.to_thread(stream.read, self.CHUNK)
            await self.ws.send(
                json.dumps(
                    {
                        "realtime_input": {
                            "media_chunks": [
                                {
                                    "data": base64.b64encode(data).decode(),
                                    "mime_type": "audio/pcm",
                                }
                            ]
                        }
                    }
                )
            )

    async def recv_model_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT, channels=self.CHANNELS, rate=24000, output=True
        )
        async for msg in self.ws:
            response = json.loads(msg)
            try:
                audio_data = response["serverContent"]["modelTurn"]["parts"][0][
                    "inlineData"
                ]["data"]
                await asyncio.to_thread(stream.write, base64.b64decode(audio_data))
            except KeyError:
                pass


if __name__ == "__main__":
    client = SimpleGeminiVoice()
    asyncio.run(client.start())
