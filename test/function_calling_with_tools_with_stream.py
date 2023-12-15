from openai import AzureOpenAI
import os, json

from src.ClsChatBot import ChatRobot

env_path = os.path.abspath('.')
chatbot = ChatRobot()
chatbot.setup_env("../key.txt", "../config.json")
client = chatbot.initial_llm()


# Example function hard coded to return the same weather
# In production, this could be your backend API or an external API
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


def run_conversation(client_input, messages_input, model_name="gpt-4-turbo"):
    # Step 1: send the conversation and available functions to the model
    client = client_input
    messages = messages_input
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_img_by_dalle3",
                "description": "Create image by call to Dall-E3 with prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The description of image to be created, e.g. a cute panda",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
        stream=True
    )
    full_response = ""
    tool_calls = []
    cur_func_call = {"id": None, "type": "function", "function": {"arguments": "", "name": None}}
    for chunk in response:
        deltas = chunk.choices[0].delta
        if deltas.content is not None:
            full_response += deltas.content  # ["answer"]  # .choices[0].delta.get("content", "")
        if deltas.tool_calls is not None:
            if deltas.tool_calls[0].id is not None:
                if cur_func_call["id"] is not None:
                    if cur_func_call["id"] != deltas.tool_calls[0].id:
                        tool_calls.append(cur_func_call)
                        cur_func_call = {"id": None, "type": "function", "function": {"arguments": "", "name": None}}
                cur_func_call["id"] = deltas.tool_calls[0].id
            if deltas.tool_calls[0].function.name is not None:
                cur_func_call["function"]["name"] = deltas.tool_calls[0].function.name
            if deltas.tool_calls[0].function.arguments is not None:
                cur_func_call["function"]["arguments"] += deltas.tool_calls[0].function.arguments
        if chunk.choices[0].finish_reason == "tool_calls":
            tool_calls.append(cur_func_call)
            cur_func_call = {"name": None, "arguments": "", "id": None}
            # function call here using func_call
            print("call tool here")
            response_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
            "create_img_by_dalle3": create_img_by_dalle3,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_response = execute_function_call(available_functions, tool_call)
            messages.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content
    else:
        return full_response


messages = [{"role": "user", "content": "请画一张可口的苹果"}] #"What's the weather like in San Francisco, Tokyo, and Paris?"
print(run_conversation(client, messages))
