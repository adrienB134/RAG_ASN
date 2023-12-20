"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from langchain.vectorstores.vectara import Vectara
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
)

from app.callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from app.query_data import get_chain
from app.schemas import ChatResponse


def startup_event():
    logging.info("loading vectorstore")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HF_TOKEN"],
        api_url="https://pikmjtam1n1c2rzu.us-east-1.aws.endpoints.huggingface.cloud",
        model_name="WhereIsAI/UAE-Large-V1",
    )

    vectorstore = Chroma(
        collection_name="ASN",
        persist_directory="./Ingestion/chroma_db",
        embedding_function=embeddings,
    )
    print("There are", vectorstore._collection.count(), "in the collection")
    return vectorstore


app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
load_dotenv()
vectorstore = startup_event()


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    chat_history.append(("", "Bonjour, je suis ChatASN votre assistant de recherche."))
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            test_db = vectorstore.similarity_search(question)
            print("TESTDB")
            print(test_db)
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())
            print(resp)

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            print("START")
            print(start_resp)
            print(f"invoke {question}")
            result = qa_chain.invoke(question)
            print("QUESTION")
            print(result)

            # chat_history.append((question, result["answer"]))
            result_resp = ChatResponse(
                sender="bot",
                message=result["answer"],
                type="stream",
            )
            await websocket.send_json(result_resp.dict())
            print("RESULT ")
            print(result_resp)

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
            print("END")
            print(end_resp)

            for document in result["documents"]:
                sources_resp = ChatResponse(
                    sender="bot",
                    message=document["source"].split("/")[-1],
                    type="sources",
                )
                await websocket.send_json(sources_resp.dict())
                print(document)
                print("SOURCES")
                print(sources_resp)

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
