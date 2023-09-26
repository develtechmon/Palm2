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

def chunking():
    loader = PyPDFLoader(r"D:/AI_CTS/Palm2/palm2_projects/palm2_pdf_chatbot_chromadb/Hotline_Wiki_v3.pdf")

    tex_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        length_function = len,
    )
    pages = loader.load_and_split(tex_splitter)
    return pages

def vector_storing(pages):
    # Save db to disk
    db = Chroma.from_documents(pages, embeddings, persist_directory=Path + "/content/db")
    
    retriever = db.as_retriever()
    return retriever

def ingest():
    doc = chunking()
    db = vector_storing(doc)

if __name__ == "__main__":
    ingest()