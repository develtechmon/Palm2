from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
import textwrap
import pprint

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

os.environ['GOOGLE_API_KEY'] = "AIzaSyCqeMc0E7WcuMmRHotLGfuIDdS-TT5zTso"

"""
Here, the instructions is part of the system prompt
Here the AI will remember previous conversation history due to `ConversationoBufferMemory
"""

template = """
Your name is Kelly. Always introduce yourself as Kelly.
You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
{chat_history}

Question: {user_input}

"""

print(template)

def init_memory():
    memory = ConversationBufferMemory(
        input_key="user_input",
        memory_key="chat_history",
    )
    return memory

prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
memory = init_memory()

llm = GooglePalm(model_name="models/text-bison-001", temperature=0.1)

LLM_Chain=LLMChain(prompt=prompt, memory=memory, llm=llm)
while True:
        query = input(f"\nPrompt >> " )
        if query == "exit":
                print("exiting")
                break
        if query == "":
                continue
        res = LLM_Chain.run(query)
        #pprint(res)
        wrapped_text = textwrap.fill(res, width=100)
        print(wrapped_text + "\n\n")




