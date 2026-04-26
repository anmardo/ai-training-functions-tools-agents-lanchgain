import os, openai, json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

import json

#way to use functions:
# if model_calls_function:
#     function_result = execute_function()
#     return call_model_again(function_result)
# else:
#    return model_response

# important: functions and params are considered as tokens in our call


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# define a function
functions = [
    {
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
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]

# Call the ChatCompletion endpoint
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions
)

print(response)

#default behavior : in this case it will be up to the model to decide
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto",
)
print(response)

#in this case it is ok not call function since message is not related
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",#force not to call function at all
)
print(response)

#in this call, since we have forced the llm to reply , it will skip our function and the model will reply
messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)

#now it will work, we are forcing it to use our specific function
messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)

#funny case, we are forcing it to use the function but the message makes no sense so it will simply return the info of the function without any processing
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)

#FULL ORCHESTRATION:
# 1. CALL MODEL PASSING FUNCTION DEFINITION

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)  #this guy will return an object "choices" that is a collection and it will contain our function call and the args

# 2. APPEND TO THE MESSAGES OBJECT THE RESULT OF CALLING OUR FUNCTION
#get the message
#before-> messages=[{'role': 'user', 'content': "What's the weather like in Boston!"}]
messages.append(response["choices"][0]["message"])
#after -> messages=
# [{'role': 'user', 'content': "What's the weather like in Boston!"}, <OpenAIObject at 0x7fb8c40bee00> JSON: {
# "role": "assistant",
# "content": null,
# "function_call": {
#     "name": "get_current_weather",
#     "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
# },
# "refusal": null,
# "annotations": []
# }]

#get the args
args = json.loads(response["choices"][0]["message"]['function_call']['arguments'])
#call the local function
observation = get_current_weather(args)
#append the result to the messages
messages.append(
    {
        "role": "function",
        "name": "get_current_weather",
        "content": observation,
    }
)
#after-> messages:
#[{'role': 'user', 'content': "What's the weather like in Boston!"}, <OpenAIObject at 0x7fb8c40bee00> JSON: {
#"role": "assistant",
#"content": null,
#"function_call": {
#    "name": "get_current_weather",
#    "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
#},
#"refusal": null,
#"annotations": []
#}, {'role': 'function', 'name': 'get_current_weather', 'content': '{"location": {"location": "Boston", "unit": "celsius"}, "temperature": "72", "unit": "fahrenheit", "forecast": ["sunny", "windy"]}'}]


# 3. CALL THE MODEL AGAIN WITH THE UPDATED LIST OF MESSAGES
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
)
print(response)
#this will return
#"choices": [
#    {
#        "index": 0,
#        "message": {
#            "role": "assistant",
#            "content": "The current weather in Boston is sunny and windy with a temperature of 72\u00b0F (22\u00b0C).",
#            "refusal": null,
#            "annotations": []
#        },
#        "logprobs": null,
#        "finish_reason": "stop"
#    }
#]