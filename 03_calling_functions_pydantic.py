import os
import openai

from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from typing import List
from pydantic import BaseModel, Field

# 1 - NATIVE PYTHON VS PYDANTIC
# native python
class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

foo = User(name="Joe",age=32, email="joe@gmail.com")
print(foo.age)

# pydantic classes

class pUser(BaseModel):
    name: str
    age: int
    email: str

foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")
print(foo_p.name)

foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com") #type failure!

class Class(BaseModel):
    students: List[pUser]

obj = Class(
    students=[pUser(name="Jane", age=32, email="jane@gmail.com")]
)
print(obj)

# 2 - USING PYDANTIC TO DEFINE OPENAI FUNCTIONS
from langchain_core.utils.function_calling import convert_to_openai_function

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport""" #mandatory desc
    airport_code: str = Field(description="airport code to get weather for")

weather_function = convert_to_openai_function(WeatherSearch)
print(weather_function)

# this will fail because the function description is required in opeanAI function
class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")
convert_to_openai_function(WeatherSearch1)

# 3 - USING IT WITH LANGCHAIN
from langchain_openai import ChatOpenAI
model = ChatOpenAI()
print(model.invoke("What is the weather today in Valencia?", functions=[weather_function]))

#another way to do it, binding
model_with_function = model.bind(functions=[weather_function])
print(model_with_function.invoke("What is the weather today in Valencia?"))

#force to use a function to be called
model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})
print(model_with_forced_function.invoke("What is the weather today in Valencia?"))
print(model_with_forced_function.invoke("hi!"))

#model to use chains

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("user", {input})
    ]
)
chain = prompt | model_with_function
print(chain.invoke({"input":"what is the weather today in Valencia?"}))

# multiple functions

class ArtistSearch(BaseModel):
    """Call to this to get names of song from an artist"""
    artist_name : str = Field (description="name of the artist")
    n:int = Field (description = "num of results")

functions = [
    convert_to_openai_function(WeatherSearch),
    convert_to_openai_function(ArtistSearch)
]
model_with_multiple_functions = model.bind(functions = functions);
print(model_with_multiple_functions.invoke("What is the weather today in Valencia?"))
print(model_with_multiple_functions.invoke("Give me three songs by Queen"))
print(model_with_multiple_functions.invoke("hi!"))
