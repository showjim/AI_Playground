import os
import azure.cognitiveservices.speech as speechsdk
import openai

# This example requires environment variables named "OPEN_AI_KEY" and "OPEN_AI_ENDPOINT"
# Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
openai.api_key = os.environ.get('OPEN_AI_KEY')
openai.azure_endpoint = os.environ.get('OPEN_AI_ENDPOINT')
openai.api_type = 'azure'
openai.api_version = '2023-12-01-preview'

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id = 'gpt-35-turbo'

client = openai.AzureOpenAI(
    api_version="2023-12-01-preview",
    api_key=openai.api_key,
    azure_endpoint=openai.azure_endpoint
)

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'),
                                       region=os.environ.get('SPEECH_REGION')
                                       )
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

# Should be the locale for the speaker's language.
speech_config.speech_recognition_language = "zh-CN"
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# The language of the voice that responds on behalf of Azure OpenAI.
speech_config.speech_synthesis_voice_name = 'zh-CN-YunxiaNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

# tts sentence end mark
tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n"]

messages = [
    {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Please reply as short as possible. Knowledge cutoff: 2023-04."}
]


# Prompts Azure OpenAI with a request and synthesizes the response.
def ask_openai(prompt):
    messages.append({"role": "user", "content": prompt})
    print("HUMAN: " + prompt)
    # Ask Azure OpenAI in streaming way
    response = client.chat.completions.create(
        model=deployment_id,
        messages=messages,
        max_tokens=200,
        stream=True
    )
    collected_messages = []
    last_tts_request = None
    full_response = ""
    # iterate through the stream response stream
    for chunk in response:
        deltas = chunk.choices[0].delta
        if deltas.content is not None:
            chunk_message = deltas.content  # extract the message
            collected_messages.append(chunk_message)  # save the message
            if chunk_message in tts_sentence_end:  # sentence end found
                text = ''.join(collected_messages).strip()  # join the recieved message together to build a sentence
                if text != '':  # if sentence only have \n or space, we could skip
                    print(f"Speech synthesized to speaker for: {text}")
                    last_tts_request = speech_synthesizer.speak_text_async(text)
                    collected_messages.clear()
            full_response += chunk_message
    messages.append({"role": "assistant", "content": full_response})
    print("AI: " + full_response)
    if last_tts_request:
        last_tts_request.get()


# Continuously listens for speech input to recognize and send as text to Azure OpenAI
def chat_with_open_ai():
    while True:
        print("Azure OpenAI is listening. Say '停止对话' or press Ctrl-Z to end the conversation.")
        try:
            # Get audio from the microphone and then send it to the TTS service.
            speech_recognition_result = speech_recognizer.recognize_once_async().get()

            # If speech is recognized, send it to Azure OpenAI and listen for the response.
            if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if speech_recognition_result.text == "停止对话。":
                    print("Conversation ended.")
                    break
                print("Recognized speech: {}".format(speech_recognition_result.text))
                ask_openai(speech_recognition_result.text)
            elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
                # break
            elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech_recognition_result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
        except EOFError:
            break


# Main

try:
    chat_with_open_ai()
except Exception as err:
    print("Encountered exception. {}".format(err))
