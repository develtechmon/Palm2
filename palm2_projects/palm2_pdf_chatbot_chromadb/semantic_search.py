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

def main(query):
    # load db from disk
    db = Chroma(persist_directory=Path + "/content/db", embedding_function=embeddings)
    
    search = db.similarity_search(query, k=3)
    print(search)

if __name__ == "__main__":
    query = input(f"\n\nPrompt: " )
    main(query)