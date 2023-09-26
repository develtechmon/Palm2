from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
import streamlit as st

import os
import textwrap

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from streamlit_chat import message

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

os.environ['GOOGLE_API_KEY'] = "AIzaSyCqeMc0E7WcuMmRHotLGfuIDdS-TT5zTso"

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

template = """
Your name is Kelly. Always introduce yourself as Kelly.
You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
{chat_history}

Question: {user_input}

"""

print(template)

# @st.cache_data(experimental_allow_widgets=True)  # 👈 Set the parameter
# def chat_model():
#     llm = GooglePalm(temperature=0.1)
#     return llm

memory = ConversationBufferMemory(input_key="user_input", memory_key="chat_history",)
main_prompt = PromptTemplate(input_variables=['chat_history','user_input'], template=template)
llm = GooglePalm(temperature=0.1)

LLM_Chain = LLMChain(
    llm=llm,
    prompt=main_prompt,
    verbose=False,
    memory=memory
)

with st.sidebar:
    st.title("Palm2")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')
    # os.environ["TOGETHER_API_KEY"] = "4ed1cb4bc5e717fef94c588dd40c5617616f96832e541351195d3fb983ee6cb5"

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

    # st.subheader("Models and Parameters")
    # select_model = st.selectbox("Choose AI model", ['Llama 27b', 'Llama 2 13b','Llama 2 70b'], key='select_model')

    # if select_model == 'Llama 2 7b':
    #     model = "togethercomputer/llama-2-7b-chat"
    
    # elif select_model == "Llama 2 13 b":
    #     model = "togethercomputer/llama-2-13b-chat"

    # elif select_model == "Llama 2 70b":
    #     model = "togethercomputer/llama-2-7b-chat"
    
if "messages" not in st.session_state.keys():
   st.session_state.messages=[{"role": "assistant","content": "How may i assist you today sir ?"}]

for message in st.session_state.messages:
   with st.chat_message(message['role']):
       st.write(message['content'])

#Clear the chat messages
def clear_chat_history():
   st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

#Clear the chat messages
st.sidebar.button('Clear Chat History', on_click=clear_chat_history())


# User provided prompt
if prompt := st.chat_input(disabled= not add_replicate_api):
    st.session_state.messages.append({"role":"user", "content":prompt})  
    with st.chat_message("user"):
        st.write(prompt)   

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            #response = LLM_Chain.run(prompt)
            response = LLM_Chain.predict(user_input=prompt)
            placeholder = st.empty()
            full_response=''
            for item in response:
                full_response+=item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role":"assistant", "content": full_response}
    st.session_state.messages.append(message)

