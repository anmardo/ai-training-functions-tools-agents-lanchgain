import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# ===================
# 1. SIMPLE CHAIN
# ===================

prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print("1-simple chain:")
print(chain.invoke({"topic": "bears"}))
print("\n")


# ===================
# 2. COMPLEX CHAIN
# ===================

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
# this will return the proper answer
print("2-complex chain")
print(retriever.invoke("where did harrison work?"))
print("\n")


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain_core.runnables import RunnableMap

# this will return the answer directly
chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser
print("2-complex chain , direct answer")
print(chain.invoke({"question":"where did harrison work"}))
print("\n")


# if we don't put into the expression, then we get whole info
inputs = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
})
# this will return
# {'context': [Document(page_content='harrison worked at kensho'),
#   Document(page_content='bears like to eat honey')],
#  'question': 'where did harrison work?'}
print("2-complex chain without filtering the answer")
print(inputs.invoke({"question": "where did harrison work?"}))
print("\n")

# ===================
# 3. BINDS
# ===================

functions = [
    {
        "name": "weather_search",
        "description": "Search for weather given an airport code",
        "parameters": {
            "type": "object",
            "properties": {
                "airport_code": {
                    "type": "string",
                    "description": "The airport code to get the weather for"
                },
            },
            "required": ["airport_code"]
        }
    }
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)
runnable = prompt | model
# this will return the message saying that we need to call the function
print("3- binding function to see if it detects that it needs to go to that function")
print(runnable.invoke({"input": "what is the weather in san francisco"}))
print("\n")

# ===================
# 4. FALLBACKS
# ===================
from langchain_openai import OpenAI
import json

print("4-fallbacks")
simple_model = OpenAI(
    temperature=0,
    max_tokens=1000,
    model="gpt-3.5-turbo-instruct"
)

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"
print("4-chaing without failing")
print(simple_model.invoke(challenge))
print("\n")

# now let's make it fail (simple_chain expects a json but it receives binary)
simple_chain = simple_model | json.loads
print("4-chain that will fail")
# uncomment to test, if not the program stops here
# print(simple_chain.invoke(challenge))
print("\n")

# now we create a chain that uses the model and parses to string before loading as json
model = ChatOpenAI(temperature=0)
structured_chain = model | StrOutputParser() | json.loads
print("4-chain that it is not going to fail since it parses before")
structured_chain.invoke(challenge)
print("\n")

# now we use the structured_chain directly as fallback for the inital simple_chain
final_chain = simple_chain.with_fallbacks([structured_chain])
print("4-chain that it is not failing since it uses previous chain as fallback")
final_chain.invoke(challenge)
print("\n")

# ===================
# 5. INTERFACES
# ===================
print("5-interfaces")
prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

iface_chain = prompt | model | output_parser

# this works as expected
print("5-interface working")
iface_chain.invoke({"topic": "bears"})
print("\n")

# this executes both in parallel (as much as possible!)
print("5-batch of executions")
print(iface_chain.batch([{"topic": "bears"}, {"topic": "frogs"}]))
print("\n")

# await can be outside an async function in python, so to use this
# it is required to wrap it into a function and call it with asyncio.run
# response = await chain.ainvoke({"topic": "bears"})
# response
