import os
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# ------------------------------------
# TOOLS
# ------------------------------------

# 1. CREATING OWN SEARCH TOOL
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"

print(search.name)
print(search.description)
print(search.args)

# MAKING THE INPUT SEARCH TO BE BASED ON A CLASS
from pydantic import BaseModel, Field
class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")

@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"

print(search.name)
print(search.description)
print(search.args)

# 3. REAL WORLD EXAMPLE
import requests
from pydantic import BaseModel, Field
import datetime

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'


print(get_current_temperature.name)
print(get_current_temperature.description)
print(get_current_temperature.args)

# convert it to an openai function syntax
from langchain_core.utils.function_calling import convert_to_openai_function
print(convert_to_openai_function(get_current_temperature))
#print(get_current_temperature.invoke({"latitude" : 13, "longitude" : 14}))

# another example
import wikipedia, self

@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
                self.wiki_client.exceptions.PageError,
                self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

print(search_wikipedia.name)
print(search_wikipedia.description)
print(convert_to_openai_function(search_wikipedia))
#print(search_wikipedia.invoke({"query" : "langchain"}))

# 4. CALLING OPENAI API USING FUNCTIONS
from langchain_community.utilities.openapi import OpenAPISpec

text = """
{
  "openapi": "3.0.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""
import json
spec = json.loads(text)

# openapi_spec_to_openai_fn does no longer exists, we use the one created as utils
# but this is only an educational purpose, it doesn't support POST calls for example
from openapi_spec_to_openai import *
pet_openai_functions, pet_callables = openapi_spec_to_tools(text)

print(pet_openai_functions)

from langchain_openai import ChatOpenAI
model = ChatOpenAI(temperature=0).bind_tools(pet_openai_functions)
# the idea is that now this model will understand which one of the apis from the spec should be
# call in order to reply to these questions
print(model.invoke("what are three pet names"))
print(model.invoke("tell me about pet with id 42"))

# ------------------------------------
# ROUTING
# ------------------------------------

functions = [
    convert_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]
model = ChatOpenAI(temperature=0).bind(functions=functions)
# this is expected to return same behavior than before (it will choose the right function every time)
print(model.invoke("what is the weather in Valencia now"))
print(model.invoke("what is langchain"))

#adding a prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are a helpful but sassy assistant"),
    ("user", "{input}")
])
chain = prompt | model
# same response is expected
print(chain.invoke({"input" : "what is the weather in Valencia now"}))

# now we are creating a function that will decide whether to call a function or not depending on the
# thype of the result
from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage

def route(result):
    # Old-style parser result
    if isinstance(result, AgentFinish):
        return result.return_values["output"]

    # Modern ChatOpenAI tool-call result
    if isinstance(result, AIMessage):
        if not result.tool_calls:
            return result.content

        tools = {
            "search_wikipedia": search_wikipedia,
            "get_current_temperature": get_current_temperature,
        }

        tool_call = result.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        return tools[tool_name].invoke(tool_args)

    raise ValueError(f"Unsupported result type: {type(result)}")

chain = prompt | model | route

print(chain.invoke ({"input" : "what is the weather in Valencia right now?"}))
print(chain.invoke ({"input" : "what is langchain"}))
print(chain.invoke ({"input" : "hi!"}))