import streamlit as st
import os, time, json
from typing import List
import azure.cognitiveservices.speech as speechsdk
from src.ClsChatBot import ChatRobot

# __version__ = "Beta V0.0.2"
env_path = os.path.abspath('.')

chatbot = ChatRobot()
chatbot.setup_env()
client = chatbot.initial_llm()
tools = chatbot.initial_tools()

# This requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'),
                                       region=os.environ.get('SPEECH_REGION'))
st.set_page_config(page_title="FreeChat - Chatbot With Native APIs")


def set_reload_mode():
    st.session_state["FreeChatReloadMode"] = True


def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["FreeChatReloadFlag"] = True


def text_2_speech(text: str, voice_name: str):
    # The language of the voice that speaks.
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    if voice_name == "None":
        voice_name = "zh-CN-XiaoyouNeural"  # "zh-CN-XiaoyiNeural"
    speech_config.speech_synthesis_voice_name = voice_name  # "zh-CN-XiaoyiNeural"  # "zh-CN-YunxiaNeural"
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


def speech_2_text():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    result_txt = ""
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        result_txt = speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        result_txt = "No speech could be recognized"
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
        result_txt = "Speech Recognition canceled"
    return result_txt


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def create_img_by_dalle3(prompt):
    """Create image by call to Dall-E3"""
    result = client.images.generate(
        model="Dalle3",  # the name of your DALL-E 3 deployment
        prompt=prompt,  # "a close-up of a bear walking through the forest",
        size='1024x1024',
        style="vivid",  # "vivid", "natural"
        quality="hd",  # "standard" "hd"
        n=1
    )
    json_response = json.loads(result.model_dump_json())
    # Retrieve the generated image
    image_url = json_response["data"][0]["url"]  # extract image URL from response
    revised_prompt = json_response["data"][0]["revised_prompt"]
    print("Dall-E3: " + revised_prompt)
    print("Dall-E3: " + image_url)
    return image_url


def execute_function_call(available_functions, tool_call):
    function_name = tool_call["function"]["name"]
    function_to_call = available_functions.get(function_name, None)
    if function_to_call:
        function_args = json.loads(tool_call["function"]["arguments"])
        function_response = function_to_call(**function_args)
    else:
        function_response = f"Error: function {function_name} does not exist"
    return function_response


def control_msg_hsitory_szie(msglist: List, max_cnt=10):
    while len(msglist) > max_cnt:
        msglist.pop(1)
    return msglist


def main():
    index = 0
    st.title('🎙️Free Chat Web-UI App')
    # Sidebar contents
    if "FreeChatReloadMode" not in st.session_state:
        st.session_state["FreeChatReloadMode"] = True
    if "FreeChatReloadFlag" not in st.session_state:
        st.session_state["FreeChatReloadFlag"] = True
    # Initialize chat history
    if "FreeChatMessages" not in st.session_state:
        st.session_state['FreeChatMessages'] = []
    if "FreeChatMessagesDispay" not in st.session_state:
        # this is a shadow of "FreeChatMessages" to keep image URL from Dalle3
        st.session_state['FreeChatMessagesDispay'] = []
    # chain = chatbot.initial_llm()
    if "FreeChatChain" not in st.session_state:
        # client = chatbot.initial_llm()
        st.session_state["FreeChatChain"] = client

    with st.sidebar:
        st.sidebar.expander("Settings")
        st.sidebar.subheader("Parameter for Chatbot")
        aa_chat_mode = st.sidebar.selectbox(label="`0. Chat Mode`",
                                            options=["CasualChat", "Translate", "西瓜一家-小南瓜", "西瓜一家-小东瓜",
                                                     "西瓜一家-Ana"],
                                            index=0,
                                            on_change=set_reload_mode)
        aa_llm_model = st.sidebar.selectbox(label="`1. LLM Model`",
                                            options=["gpt-35-turbo", "gpt-35-turbo-16k", "gpt-4", "gpt-4-turbo"],
                                            index=0,
                                            on_change=set_reload_flag)
        aa_temperature = st.sidebar.selectbox(label="`2. Temperature (0~1)`",
                                              options=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                              index=1,
                                              on_change=set_reload_flag)
        if "16k" in aa_llm_model:
            aa_max_resp_max_val = 16 * 1024
        else:
            aa_max_resp_max_val = 4096
        aa_max_resp = st.sidebar.slider(label="`3. Max response`",
                                        min_value=256,
                                        max_value=aa_max_resp_max_val,
                                        value=256,
                                        on_change=set_reload_flag)
        aa_context_msg = st.sidebar.slider(label="`4. Context message`",
                                           min_value=10,
                                           max_value=100,
                                           value=20,
                                           on_change=set_reload_flag)

        if st.session_state["FreeChatReloadMode"] == True:
            system_prompt = chatbot.select_chat_mode(aa_chat_mode)
            st.session_state['FreeChatMessages'] = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "I'm FreeChatBot, How may I help you?"}
            ]
            st.session_state['FreeChatMessagesDisplay'] = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "I'm FreeChatBot, How may I help you?"}
            ]
            st.session_state["FreeChatReloadMode"] = False
        if st.session_state["FreeChatReloadFlag"] == True:
            if "FreeChatSetting" not in st.session_state:
                st.session_state["FreeChatSetting"] = {}
            st.session_state["FreeChatSetting"] = {"model": aa_llm_model, "max_tokens": aa_max_resp,
                                                   "temperature": float(aa_temperature), "context_msg": aa_context_msg}
            st.session_state["FreeChatReloadFlag"] = False

        # Text2Speech
        aa_voice_name = st.sidebar.selectbox(label="`5. Voice Name`",
                                             options=["None", "小南瓜", "小东瓜", "Ana"],
                                             index=0)
        speech_config.speech_recognition_language = "zh-CN"  # "zh-CN" #"en-US"
        if aa_voice_name == "小南瓜":
            aa_voice_name = "zh-CN-XiaoyiNeural"
        elif aa_voice_name == "小东瓜":
            aa_voice_name = "zh-CN-YunxiaNeural"
        elif aa_voice_name == "Ana":
            aa_voice_name = "en-US-AnaNeural"
            speech_config.speech_recognition_language = "en-US"  # "zh-CN" #"en-US"

        # Speech2Text
        speech_txt = ""
        if st.sidebar.button("`Speak`"):
            speech_txt = speech_2_text()

    # Display chat messages from history on app rerun
    for message in st.session_state["FreeChatMessagesDisplay"]:
        if message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            if message["content"] is not None:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    st.button(label="Play", key="history" + str(index), on_click=text_2_speech,
                              args=(message["content"], aa_voice_name,))
                    index += 1
                    if "image" in message.keys():
                        st.image(message["image"], width=256)

    # Accept user input
    if (prompt := st.chat_input("Type you input here")) or (prompt := speech_txt):
        # Add user message to chat history
        max_cnt = st.session_state["FreeChatSetting"]["context_msg"]
        st.session_state["FreeChatMessages"] = control_msg_hsitory_szie(st.session_state["FreeChatMessages"], max_cnt)
        st.session_state["FreeChatMessages"].append({"role": "user", "content": prompt})
        st.session_state["FreeChatMessagesDisplay"].append({"role": "user", "content": prompt})

        print("HUMAN: " + prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            btn_placeholder = st.empty()
            full_response = ""
            function_response = ""
            tool_calls = []
            cur_func_call = {"id": None, "type": "function", "function": {"arguments": "", "name": None}}
            with st.spinner('preparing answer'):  # st.session_state["FreeChatChain"]
                response = st.session_state["FreeChatChain"].chat.completions.create(
                    model=st.session_state["FreeChatSetting"]["model"],
                    messages=st.session_state['FreeChatMessages'],
                    max_tokens=st.session_state["FreeChatSetting"]["max_tokens"],
                    # default max tokens is low so set higher
                    temperature=st.session_state["FreeChatSetting"]["temperature"],
                    stream=True,
                    tools=tools,
                    tool_choice="auto",  # auto is default, but we'll be explicit
                )
                for chunk in response:
                    # process normal response and tool_calls response
                    deltas = chunk.choices[0].delta
                    if deltas.content is not None:
                        full_response += deltas.content  # ["answer"]  # .choices[0].delta.get("content", "")
                        time.sleep(0.001)
                        message_placeholder.markdown(full_response + "▌")
                    elif deltas.tool_calls is not None:
                        if deltas.tool_calls[0].id is not None:
                            if cur_func_call["id"] is not None:
                                if cur_func_call["id"] != deltas.tool_calls[0].id:
                                    tool_calls.append(cur_func_call)
                                    cur_func_call = {"id": None, "type": "function",
                                                     "function": {"arguments": "", "name": None}}
                            cur_func_call["id"] = deltas.tool_calls[0].id
                        if deltas.tool_calls[0].function.name is not None:
                            cur_func_call["function"]["name"] = deltas.tool_calls[0].function.name
                        if deltas.tool_calls[0].function.arguments is not None:
                            cur_func_call["function"]["arguments"] += deltas.tool_calls[0].function.arguments
                    elif chunk.choices[0].finish_reason == "tool_calls":
                        tool_calls.append(cur_func_call)
                        cur_func_call = {"name": None, "arguments": "", "id": None}
                        # function call here using func_call
                        # print("call tool here")
                        response_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}

                # Step 2: check if the model wanted to call a function
                if tool_calls:
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors
                    available_functions = {
                        "get_current_weather": get_current_weather,
                        "create_img_by_dalle3": create_img_by_dalle3,
                    }  # only one function in this example, but you can have multiple
                    # extend conversation with assistant's reply
                    st.session_state['FreeChatMessages'].append(response_message)
                    st.session_state['FreeChatMessagesDisplay'].append(response_message)
                    # Step 4: send the info for each function call and function response to the model
                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        function_response = execute_function_call(available_functions, tool_call)
                        st.session_state['FreeChatMessages'].append(
                            {
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )  # extend conversation with function response
                        st.session_state['FreeChatMessagesDisplay'].append(
                            {
                                "tool_call_id": tool_call["id"],
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )  # extend conversation with function response
                    second_response = st.session_state["FreeChatChain"].chat.completions.create(
                        model=st.session_state["FreeChatSetting"]["model"],
                        messages=st.session_state['FreeChatMessages'],
                        max_tokens=st.session_state["FreeChatSetting"]["max_tokens"],
                        # default max tokens is low so set higher
                        temperature=st.session_state["FreeChatSetting"]["temperature"],
                        stream=True,
                    )  # get a new response from the model where it can see the function response
                    for chunk in second_response:
                        deltas = chunk.choices[0].delta
                        if deltas.content is not None:
                            full_response += deltas.content  # ["answer"]  # .choices[0].delta.get("content", "")
                            time.sleep(0.001)
                            message_placeholder.markdown(full_response + "▌")
                else:
                    full_response = full_response
            message_placeholder.markdown(full_response)

            st.session_state['FreeChatMessages'].append({"role": "assistant", "content": full_response})
            if function_response.startswith("https://"):
                st.session_state['FreeChatMessagesDisplay'].append(
                    {"role": "assistant", "content": full_response, "image": function_response})
                st.image(function_response)
            else:
                st.session_state['FreeChatMessagesDisplay'].append(
                    {"role": "assistant", "content": full_response})
            print("AI: " + full_response)
            if aa_voice_name != "None":
                text_2_speech(full_response, aa_voice_name)
            btn_placeholder.button(label="Play", key="current", on_click=text_2_speech,
                                   args=(full_response, aa_voice_name,))


if __name__ == "__main__":
    main()
