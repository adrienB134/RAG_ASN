"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from langchain.vectorstores.vectara import Vectara
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse


# def startup_event():
#     logging.info("loading vectorstore")

#     global vectorstore
#     vectorstore = Vectara(
#         vectara_customer_id=os.environ["VECTARA_CUSTOMER_ID"],
#         vectara_corpus_id=os.environ["VECTARA_CORPUS_ID"],
#         vectara_api_key=os.environ["VECTARA_API_KEY"],
#     )


# def startup_event():
#     logging.info("loading vectorstore")
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     global vectorstore
#     vectorstore = FAISS.load_local("./Ingestion/faiss_index", embeddings)
#     test_db = vectorstore.similarity_search_with_score(
#         "la baisse de la température et de la pression du réacteur doit être réalisée dans \ndes délais contraints3, représentatifs des délais nécessaires pour procéder au diagnostic",
#         fetch_k=5,
#     )
#     print(f"There are", test_db, "in the collection")


def startup_event():
    logging.info("loading vectorstore")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        collection_name="lettres_de_suivi",
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

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
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

            for documents in [result["sources"]]:
                sources_resp = ChatResponse(
                    sender="bot",
                    message=result["sources"].split("/")[-1],
                    type="sources",
                )
                await websocket.send_json(sources_resp.dict())
                print(documents)
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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
