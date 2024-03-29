import os, sys, cv2, base64, openai
import azure.cognitiveservices.speech as speechsdk
import threading
import queue, time
# For VS Code use src module, need to nop in Pycharm
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#########################
from src.ClsChatBot import ChatRobot

# env_path = os.path.abspath(".")
chatbot = ChatRobot()
# For VS Code key.txt & config.json, in Pycharm use key.txt & config.json
chatbot.setup_env("key.txt", "config.json")
#########################
client = chatbot.initial_llm()

is_tts_speaking = False
shared_bool = threading.Event()

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id = "gpt-4-vision-preview"

# client = openai.AzureOpenAI(
#     api_version="2023-12-01-preview",
#     api_key=openai.api_key,
#     azure_endpoint=openai.azure_endpoint
# )

audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

# The language of the voice that responds on behalf of Azure OpenAI.
chatbot.speech_config.speech_synthesis_voice_name = 'zh-CN-YunxiaNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=chatbot.speech_config, audio_config=audio_output_config)

# tts sentence end mark
tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n"]

# messages = [
#     {"role": "system", "content": "请使用中文回答。These are frames from a video that I want you to talk about with. Knowledge cutoff: 2023-04."}
# ]


def encode_image_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode('utf-8')

def ask_to_gpt(base64Frames, prompt):
    messages = [
        {"role": "system", "content": "These are frames from a video. Please reply in Chinese, according to the frames."}
    ]
    print("HUMAN: " + prompt)
    PROMPT_MESSAGES = {
            "role": "user",
            "content": [
                prompt, #"These are frames from a video that I want to upload. Please describe what you see.",
                *map(lambda x: {"image": x, "resize": 480}, base64Frames[0::30]),
            ],
    }
    messages.append(PROMPT_MESSAGES)
    params = {
        "model": deployment_id,
        "messages": messages,
        "max_tokens": 200,
        "stream": True,
    }
    try:
        response = client.chat.completions.create(**params)

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
                        last_tts_request = speech_synthesizer.speak_text(text)
                        collected_messages.clear()
                full_response += chunk_message
        messages.append({"role": "assistant", "content": full_response})
        print("AI: " + full_response)
        # if last_tts_request:
        #     last_tts_request.get()
    except Exception as e:
        print(f"An error occurred: {e}")
        # 可以选择在这里返回，或者处理错误后继续
        return e
    
    return full_response

def gpt_thread(base64Frames, prompt):
    global is_tts_speaking
    global shared_bool
    # 设置标志，以便音频识别线程知道TTS正在发声
    is_tts_speaking = True
    shared_bool.set()
    # 调用 GPT 函数并处理结果
    response_str = ask_to_gpt(base64Frames, prompt)
    # TTS发声结束后清除标志
    is_tts_speaking = False
    shared_bool.clear()

def clear_audio_queue(audio_queue):
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()  # 从队列中移除并丢弃所有项目
        except queue.Empty:
            continue
        audio_queue.task_done()  # 通知任务完成

def audio_recognition_thread(audio_queue):
    global is_tts_speaking
    global shared_bool
    while True:
        # Should be the locale for the speaker's language.
        chatbot.speech_config.speech_recognition_language = "zh-CN"
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=chatbot.speech_config, audio_config=audio_config)
        speech_result_str = ""
        if not shared_bool.is_set(): #not is_tts_speaking:
            try:
                # 获取音频识别结果
                speech_recognition_result = speech_recognizer.recognize_once_async().get()
                
                if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    speech_result_str = speech_recognition_result.text
                    if speech_result_str == "停止对话。":
                        print("Conversation ended.")
                        break
                    # 将结果放入队列
                    audio_queue.put(speech_result_str)
                    print("Recognized speech: {}".format(speech_result_str))
                elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                    print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
                    # break
                elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speech_recognition_result.cancellation_details
                    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        print("Error details: {}".format(cancellation_details.error_details))
            except Exception as e:
                print(f"An error occurred during speech recognition: {e}")
                # 可能需要重置识别器或处理错误
        else:
            # 如果TTS正在发声，我们稍等一会儿再继续
            speech_recognizer.stop_continuous_recognition()
            time.sleep(0.1)

def display_frames_thread(frame_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def main():
    global is_tts_speaking
    # 创建一个队列用于从音频识别线程传递结果
    audio_queue = queue.Queue()
    # 启动音频识别线程
    audio_thread = threading.Thread(target=audio_recognition_thread, args=(audio_queue,))
    audio_thread.start()

    # 打开摄像头设备，0通常是指内置摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置帧率
    desired_fps = 30.0  # 例如，设置为30帧每秒
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # 设置分辨率
    desired_width = 1280 #640  # 例如，设置宽度为640
    desired_height = 720 #480  # 例如，设置高度为480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    base64Frames = []

    # # 创建一个队列用于从视频捕获线程传递帧
    # frame_queue = queue.Queue()

    # # 启动显示帧的线程
    # display_thread = threading.Thread(target=display_frames_thread, args=(frame_queue,))
    # display_thread.start()
    time.sleep(1)
    try:
        print("Azure OpenAI is listening. Say '停止对话' or press Ctrl-Z to end the conversation.")
        while True:
            # 逐帧捕获
            ret, frame = cap.read()
            
            # 如果正确读取帧，ret为True
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # # 将帧放入队列
            # frame_queue.put(frame)
            # Encode the frame in Base64
            base64_image = encode_image_to_base64(frame)
            base64Frames.append(base64_image)
            
            # Get audio from the microphone and then send it to the TTS service.
            # speech_recognition_result = speech_recognizer.recognize_once_async().get()
            # 检查音频队列是否有新的识别结果
            if not audio_queue.empty():
                speech_recognition_result_txt = audio_queue.get()

                # If speech is recognized, send it to Azure OpenAI and listen for the response.
                speech_result_str = speech_recognition_result_txt
                        
                if len(base64Frames) >= 3 * 30: #store 3 second
                    base64Frames = base64Frames[-90:]
                    if speech_result_str != "":
                        # 在这里，你可以处理帧，例如发送到GPT-4 Vision进行分析
                        # 设置标志，以便音频识别线程知道TTS正在发声
                        is_tts_speaking = True
                        # 在TTS发声前清空队列
                        clear_audio_queue(audio_queue)

                        # 创建并启动 GPT 线程
                        gpt_thread_handle = threading.Thread(target=gpt_thread, args=(base64Frames, speech_result_str))
                        gpt_thread_handle.start()

                        # to store next 3 second video
                        base64Frames = []
                    else:
                        # clear the frames list
                        base64Frames = []
                

            # 显示结果帧
            cv2.imshow('Frame', frame)

            # 按'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放摄像头资源
        cap.release()
        # 等待 GPT 线程结束
        if 'gpt_thread_handle' in locals():
            gpt_thread_handle.join()
        # 等待音频线程结束
        audio_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()