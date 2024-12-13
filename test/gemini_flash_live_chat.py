import asyncio
import base64
import json
import os
import pyaudio
import numpy as np
from websockets.asyncio.client import connect


class SimpleGeminiVoice:
    def __init__(self):
        self.audio_queue = asyncio.Queue()
        self.api_key =  os.environ.get("GEMINI_API_KEY")
        self.model = "gemini-2.0-flash-exp"
        self.uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.CHUNK = 512
        self.RATE = 16000
        # Noise gate settings
        self.NOISE_THRESHOLD = 70  # Adjust this value to control sensitivity

    async def start(self):
        # Initialize websocket
        self.ws = await connect(
            self.uri, additional_headers={"Content-Type": "application/json"}
        )
        await self.ws.send(json.dumps({"setup": {"model": f"models/{self.model}"}}))
        await self.ws.recv(decode=False)
        print("Connected to Gemini, You can start talking now")
        # Start audio streaming using gather instead of TaskGroup
        await asyncio.gather(
            self.capture_audio(),
            self.stream_audio(),
            self.play_response()
        )

    async def capture_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        while True:

            # Set exception_on_overflow to False to prevent input overflow errors
            data = await asyncio.to_thread(
                stream.read, 
                self.CHUNK, 
                exception_on_overflow=False
            )
            # Convert bytes to numpy array for processing
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Calculate RMS value to determine audio level
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            # Only send audio if it's above the noise threshold
            if True: #rms > self.NOISE_THRESHOLD:
                print(f"Info: current RMS of sound is: {rms}")
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


    async def stream_audio(self):
        async for msg in self.ws:
            response = json.loads(msg)
            try:
                audio_data = response["serverContent"]["modelTurn"]["parts"][0][
                    "inlineData"
                ]["data"]
                self.audio_queue.put_nowait(base64.b64decode(audio_data))
            except KeyError:
                pass
            try:
                turn_complete = response["serverContent"]["turnComplete"]
            except KeyError:
                pass
            else:
                ## I commented this out because it was causing the audio to interruptions playing
                if turn_complete:
                    # If you interrupt the model, it sends an end_of_turn. For interruptions to work, we need to empty out the audio queue
                    print("\nEnd of turn")
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()

    async def play_response(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT, channels=self.CHANNELS, rate=24000, output=True
        )
        while True:
            data = await self.audio_queue.get()
            await asyncio.to_thread(stream.write, data)


if __name__ == "__main__":
    client = SimpleGeminiVoice()
    asyncio.run(client.start())
