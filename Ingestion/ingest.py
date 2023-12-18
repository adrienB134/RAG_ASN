"""Load PDF from files, clean up, split, ingest into a vectorstore"""
import os, logging
import time
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    replace_mime_encodings,
    replace_unicode_quotes,
)


def ingest_docs():
    load_dotenv()

    for folder in range(1, 2):
        folder_start = time.time()
        for file in os.listdir(f"ASN/lettres_de_suivi/{folder}"):
            start_time = time.time()
            attempts = 0
            success = False
            while attempts < 3 and not success:
                try:
                    print(" ")
                    print(f"File: {file}, Attempt n° {attempts}")
                    loader = UnstructuredFileLoader(
                        f"ASN/lettres_de_suivi/{folder}/{file}",
                        # mode="elements",
                        post_processors=[
                            clean_extra_whitespace,
                            replace_unicode_quotes,
                            replace_mime_encodings,
                        ],
                        # strategy="ocr_only",
                    )
                    raw_documents = loader.load()

                    unrecognized_char = False
                    for doc in raw_documents:
                        for char in doc.page_content:
                            if char == "\uFFFD":
                                unrecognized_char = True
                                print(f"{char}")
                                break

                    if unrecognized_char:
                        print(f"Unrecognized characters, switching to OCR.")
                        loader = UnstructuredFileLoader(
                            f"ASN/lettres_de_suivi/{folder}/{file}",
                            # mode="elements",
                            post_processors=[
                                clean_extra_whitespace,
                                replace_unicode_quotes,
                                replace_mime_encodings,
                            ],
                            strategy="ocr_only",
                        )
                        raw_documents = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200,
                    )
                    documents = text_splitter.split_documents(raw_documents)

                    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
                        api_key=os.environ["HF_TOKEN"],
                        api_url="https://pikmjtam1n1c2rzu.us-east-1.aws.endpoints.huggingface.cloud",
                        model_name="WhereIsAI/UAE-Large-V1",
                    )

                    for i in range(0, len(documents), 32):
                        vectorstore_db = Chroma.from_documents(
                            documents,
                            embeddings_model,
                            persist_directory="./chroma_db",
                            collection_name="lettres_de_suivi",
                        )

                    print(
                        f"Success! File {file} done in {round(time.time() - start_time,2)} s!"
                    )
                    success = True

                except Exception as exception:
                    logging.warning(f"Failure n° {attempts}, {exception}")
                    attempts += 1
                    if attempts == 3:
                        with open("failures_list.txt", "a") as f:
                            f.write(f"{file}\n")
                        break

        print(" ---- ")
        print(f"Folder {folder} done in {round((time.time() - folder_start)/60, 2)}min")

    print("There are", vectorstore_db._collection.count(), "in the collection")


if __name__ == "__main__":
    ingest_docs()
