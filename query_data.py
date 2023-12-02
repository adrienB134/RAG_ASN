"""Create a LangChain chain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Vectara
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# from ingest import docsearch


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> RetrievalQAWithSourcesChain:
    """Create a chain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    load_dotenv()
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    docsearch = vectorstore

    # question_gen_llm = ChatOpenAI(
    #     model="gpt-3.5-turbo-16k-0613",
    #     temperature=0,
    #     verbose=True,
    #     callback_manager=question_manager,
    # )
    # streaming_llm = ChatOpenAI(
    #     streaming=True,
    #     callback_manager=stream_manager,
    #     verbose=True,
    #     temperature=0,
    #     model="gpt-3.5-turbo-16k-0613",
    # )

    # question_generator = LLMChain(
    #     llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    # )
    # doc_chain = load_qa_chain(
    #     streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    # )

    # qa = ConversationalRetrievalChain(
    #     retriever=docsearch.as_retriever(),
    #     combine_docs_chain=doc_chain,
    #     question_generator=question_generator,
    #     callback_manager=manager,
    #     return_source_documents=True,
    # )

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # qa = ConversationalRetrievalChain.from_llm(
    #     ChatOpenAI(
    #         model="gpt-3.5-turbo-16k-0613",
    #         temperature=0,
    #         streaming=True,
    #     ),
    #     docsearch.as_retriever(),
    #     # memory=memory,
    #     callback_manager=manager,
    #     return_source_documents=True,
    # )

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0,
            streaming=True,
        ),
        chain_type="map_reduce",
        retriever=docsearch.as_retriever(),
    )
    return qa
