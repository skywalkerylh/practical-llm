# imports

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

class Config:
    BASE_URL = "http://localhost:11434/v1"
    API_KEY = "nokeyneeded"
    DEFAULT_MODEL = "llama3.1:8b"
    DB_NAME = "vector_db"


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def load_documents(self, folders: list[str]) -> list:

        documents = []
        for folder in folders:
            print(folder)
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            folder_docs = loader.load()
            documents.extend([self._add_metadata(doc, doc_type) for doc in folder_docs])

        self.documents = documents
        return documents

    def split_documents(self) -> list:
        if not self.documents:
            raise ValueError("No documents loaded. Please call load_documents first.")

        return self.text_splitter.split_documents(self.documents)

    def _add_metadata(self, doc, doc_type: str):
        doc.metadata["doc_type"] = doc_type
        return doc
    
    @staticmethod
    def create_embeddings(chunks: list[Document]):
        # model 
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # create vectordb
        if os.path.exists(Config.DB_NAME):
            Chroma(
                persist_directory=Config.DB_NAME, embedding_function=embeddings
            ).delete_collection()

        # Convert doc text into vectors 
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=Config.DB_NAME
        )
        print(f"Vectorstore created with {vectorstore._collection.count()} documents")

        return vectorstore

class Chatbot:
    def __init__(self, database):

        llm = ChatOpenAI(temperature=0.7, 
                     model_name=Config.DEFAULT_MODEL, 
                     base_url=Config.BASE_URL,
                     api_key = Config.API_KEY)

        # set up the conversation memory for the chat
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # the retriever is an abstraction over the VectorStore that will be used during RAG
        retriever = database.as_retriever()

        # putting it together: set up the conversation chain with the LLM, the vector store and memory
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, memory=memory
        )
    def chat(self, question, history):
        result = self.conversation_chain.invoke({"question": question})
        return result["answer"]
    
    def run_text_only(self, query):
        
        result = self.conversation_chain.invoke({"question": query})
        print(result["answer"])
    
    def run_UI(self):
        view = gr.ChatInterface(self.chat, type="messages").launch(inbrowser=True)


if __name__ == "__main__":

    os.chdir('week5-RAG')
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

    folders = glob.glob("knowledge-base/*")
    docs = processor.load_documents(folders)
    print(f"Loaded {len(docs)} documents")

    chunks = processor.split_documents()
    print(f"Created {len(chunks)} splits")

    vectors = processor.create_embeddings(chunks)

    Chatbot(vectors).run_UI()
    
    #query = "Please explain what Insurellm is in a couple of sentences"
    #Chatbot(vectors).run_text_only(query)
