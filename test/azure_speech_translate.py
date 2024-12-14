import os
import azure.cognitiveservices.speech as speechsdk
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-translate-speech?tabs=terminal&pivots=programming-language-python
from src.ClsChatBot import ChatRobot

env_path = os.path.abspath(".")
chatbot = ChatRobot()
chatbot.setup_env("../key.txt", "../config.json")
client = chatbot.initial_llm()

speech_key, service_region = os.environ["SPEECH_KEY"], os.environ["SPEECH_REGION"]
# 'en' https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=speech-translation
from_language, to_language = 'en-US', 'zh-Hans'


def translate_speech_to_text():
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=speech_key, region=service_region)

    translation_config.speech_recognition_language = from_language
    translation_config.add_target_language(to_language)

    translation_recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config)

    print('Say something...')
    # Start an endless loop
    while True:
        translation_recognition_result = translation_recognizer.recognize_once()
        print(get_result_text(translation_recognition_result.reason, translation_recognition_result))


def get_result_text(reason, result):
    reason_format = {
        speechsdk.ResultReason.TranslatedSpeech:
            f'RECOGNIZED "{from_language}": {result.text}\n' +
            f'TRANSLATED into "{to_language}"": {result.translations[to_language]}',
        speechsdk.ResultReason.RecognizedSpeech: f'Recognized: "{result.text}"',
        speechsdk.ResultReason.NoMatch: f'No speech could be recognized: {result.no_match_details}',
        speechsdk.ResultReason.Canceled: f'Speech Recognition canceled: {result.cancellation_details}'
    }
    return reason_format.get(reason, 'Unable to recognize speech')


translate_speech_to_text()