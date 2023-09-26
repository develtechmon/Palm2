
import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
import pandas as pd
import numpy as np
import random

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import ElasticVectorSearch
from streamlit_chat import message
from time import sleep

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
import chromadb

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import os 

os.environ['GOOGLE_API_KEY'] = "AIzaSyCqeMc0E7WcuMmRHotLGfuIDdS-TT5zTso"

Path = r"D:/AI_CTS/Palm2/palm2_projects/palm2_pdf_chatbot_chromadb/"
embeddings = GooglePalmEmbeddings()

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

custom_prompt_template = """
Your name is Kelly Virtual X-Fab Hotline Agent. Always introduce your name first in the conversation. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided.
You answer should only answer the question once and not have any text after the answer is done.\n\nIf a question does not make any sense, or is not factually
coherent, explain why instead of answering something not correct. If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.\n
CONTEXT:/n/n {context}/n
{chat_history}
Question: {question}
"""
print(custom_prompt_template)

llm = GooglePalm(temperature=0.5)

flag = 0

with st.sidebar:
    st.title("Hotline Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')
    # os.environ["TOGETHER_API_KEY"] = "4ed1cb4bc5e717fef94c588dd40c5617616f96832e541351195d3fb983ee6cb5"

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to enter your prompt message!', icon='👉')

if "messages" not in st.session_state.keys():
    st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationBufferMemory(input_key='question', memory_key='chat_history', return_messages=True, output_key='answer')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Clear the chat messages
def clear_chat_history():
    st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history())

# User provided prompt
if prompt := st.chat_input(disabled= not add_replicate_api):
    st.session_state.messages.append({"role":"user", "content":prompt})  
    with st.chat_message("user"):
        st.write(prompt)  
        flag = 1   

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'question'],
                            template=custom_prompt_template)
    return prompt

def conversationalretrieval_qa_chain(llm, prompt, db, memory):
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     chain_type= 'stuff',
                                                     retriever=db.as_retriever(search_kwargs={'k': 3}),
                                                     verbose=True,
                                                     memory=memory,
                                                     return_source_documents=True,
                                                     combine_docs_chain_kwargs=chain_type_kwargs
                                                     )
    return qa_chain

def load_db():
    db = Chroma(persist_directory=Path + "/content/db", embedding_function=embeddings)
    return db

def semantic_search(docsearch,query):
    docs=docsearch.similarity_search(query)
    return docs

# Generate new response if last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    db = load_db()
    default_prompt = set_custom_prompt()
    qa = conversationalretrieval_qa_chain(llm, default_prompt, db, st.session_state.entity_memory)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            result = qa({"question": prompt})
            similarity = semantic_search(db,prompt)

            print(result)
            response = result['answer']
            placeholder = st.empty()
            full_response=''
            for item in response:
                full_response+=item
                sleep(0.01)
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

            with st.expander("Source"):
                for i in range(len(result['source_documents'])):
                    source = result['source_documents'][i].metadata.get("source")
                    page = result['source_documents'][i].metadata.get("page")
                    #st.markdown(f"""### <span style="color:green"> {source}</span>""",unsafe_allow_html=True)
                    st.info(source)
                    #st.info()

            with st.expander("Popular answer"):
                for i in range(len(similarity)):
                    answer = similarity[i].page_content
                    st.info(answer)
                
    message = {"role":"assistant", "content": full_response}
    st.session_state.messages.append(message)

#st.markdown("<h6 style='text-align: right; color: white;'>Built by <a href='https://github.com/AIAnytime'>X-Fab with ❤️ </a></h3>", unsafe_allow_html=True)

if flag == 1:
    with st.expander("Do you find this answer is helpful ?"):
        st.markdown("For further support, please submit your request to hotline@xfab.com")
