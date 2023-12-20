import os, time
import azure.cognitiveservices.speech as speechsdk

def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription="SPEECH_KEY", region="eastasia")
    speech_config.speech_recognition_language="zh-CN" #"en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

def recognize_from_microphone_continuous():
    speech_config = speechsdk.SpeechConfig(subscription="SPEECH_KEY", region="eastasia")
    speech_config.speech_recognition_language="zh-CN"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt):
        """callback that stops continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True

    speech_recognizer.recognized.connect(lambda evt: print('Recognized: {}'.format(evt.result.text)))

    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    
    while not done:
        time.sleep(0.5)

def recognize_from_microphone_continuous2():
    speech_config = speechsdk.SpeechConfig(subscription="13b373f485be46c3a5f2639a3fc757e4", region="eastasia")
    speech_config.speech_recognition_language = "zh-CN"
    # speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1000")
    # speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1000")
    # speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1000")
    # speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1000")

    # speech_config.end_silence_timeout_ms = 1000
    # speech_config = 3000
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False
    full_text = ""  # Variable to store the full recognized text
    last_speech_time = time.time()  # Initialize the last speech time

    def recognized_cb(evt):
        nonlocal full_text
        nonlocal done
        nonlocal last_speech_time
        # Append the recognized text to the full_text variable
        full_text += evt.result.text + " "
        # Check the recognized text for the stop phrase
        print("OK")
        print('RECOGNIZED: {}'.format(evt))
        last_speech_time = time.time()  # Reset the last speech time
        if "停止录音" in evt.result.text:
            print("Stop phrase recognized, stopping continuous recognition.")
            speech_recognizer.stop_continuous_recognition_async()
            done = True

    def recognizing_cb(evt):
        # This callback can be used to show intermediate results.
        nonlocal last_speech_time
        last_speech_time = time.time()  # Reset the last speech time

    def canceled_cb(evt):
        print("Canceled: {}".format(evt.reason))
        if evt.reason == speechsdk.CancellationReason.Error:
            print("Cancellation Error Details: {}".format(evt.error_details))
        # speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True

    def stop_cb(evt):
        print('CLOSING on {}'.format(evt))
        # speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True

    # # Connect callbacks to the events fired by the speech recognizer
    # speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    # speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    # speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    # speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    # speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(canceled_cb)

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.recognizing.connect(recognizing_cb)
    # speech_recognizer.session_stopped.connect(stop_cb)
    # speech_recognizer.canceled.connect(canceled_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition_async()
    while not done:
        time.sleep(.5)  # You can also use time.sleep() to wait for a short amount of time
        if time.time() - last_speech_time > 3:  # If it's been more than 3 seconds since last speech
            print("3 seconds of silence detected, stopping continuous recognition.")
            speech_recognizer.stop_continuous_recognition_async()
            done = True

    # Stop recognition to clean up
    speech_recognizer.stop_continuous_recognition_async()

    return full_text.strip()  # Return the full text without leading/trailing spaces


a = recognize_from_microphone_continuous2()
print(a)
# recognize_from_microphone()