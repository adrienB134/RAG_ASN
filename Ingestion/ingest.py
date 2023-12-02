"""Load PDF from files, clean up, split, ingest into a vectorstore"""
import pickle
import time
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, FakeEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.vectara import Vectara
from langchain.vectorstores.chroma import Chroma


def ingest_docs():
    load_dotenv()
    for folder in range(0, 100):
        loader = PyPDFDirectoryLoader(f"ASN/lettres_de_suivi/{folder}")
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(raw_documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("faiss_index")
        print(f"Folder {folder} done")


# def ingest_docs():
#     load_dotenv()

#     for folder in range(0, 3):
#         loader = PyPDFDirectoryLoader(f"ASN/lettres_de_suivi/{folder}")
#         raw_documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=3000,
#             chunk_overlap=200,
#         )
#         documents = text_splitter.split_documents(raw_documents)
#         embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectorstore_db = Chroma.from_documents(
#             documents,
#             embeddings,
#             persist_directory="./chroma_db",
#             collection_name="lettres_de_suivi",
#         )
#         print(f"Folder {folder} done")

#     print("There are", vectorstore_db._collection.count(), "in the collection")


# def ingest_docs():
#     load_dotenv()
#     a = 0
#     for folder in range(0, 370):
#         loader = PyPDFDirectoryLoader(f"ASN/lettres_de_suivi/{folder}")
#         raw_documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=10000,
#             chunk_overlap=200,
#         )
#         documents = text_splitter.split_documents(raw_documents)
#         Vectara.from_documents(
#             documents,
#             FakeEmbeddings(size=10000),
#         )
#         print(f"Folder {folder} done")
#         time.sleep(5)


if __name__ == "__main__":
    ingest_docs()
