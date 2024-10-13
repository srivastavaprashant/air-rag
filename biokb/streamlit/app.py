import webbrowser
from pathlib import Path
from typing import Callable

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

from biokb.agent import AIRAgent
from biokb.llm import AIRLLM
from biokb.embedding import AIREmbedding, AIRPubmedSearch
from biokb.helpers import get_generation_config
from biokb.utils import get_file_names, create_documents_from_text_files, get_logger
from biokb.settings import MODEL_NAME, EMBEDDING_MODEL_NAME, DB_DIR, DATA_DIR
from biokb.tools import search_tool, response

st.set_page_config(page_title="AIR assistant", page_icon="💡")

logger = get_logger('air-webapp', debug=True)
logger.info("Starting AIR assistant...")

@st.cache_resource
def get_llm():
    return AIRLLM(
        model_name=MODEL_NAME,
        generation_config=get_generation_config(t=0.1),
        logger=logger
    )

@st.cache_resource
def get_embedding_llm():
    return AIREmbedding(
        model_name=EMBEDDING_MODEL_NAME
    )

logger.info("Loading models...")
embedding_llm = get_embedding_llm()
llm = get_llm()

if Path(DB_DIR).exists():
    docstore = AIRPubmedSearch.load(
        DB_DIR, 
        embedding_llm=embedding_llm,
    )
else:
    files = get_file_names(DATA_DIR)
    documents = create_documents_from_text_files(files)
    print(f"Number of documents: {len(documents)}")
    docstore = AIRPubmedSearch(
        documents=documents,
        embedding_llm=embedding_llm,
    )
    docstore.build(DB_DIR)


@st.cache_resource
def get_agent() -> AIRAgent:
    return AIRAgent(llm=llm, docstore=docstore, logger=logger, tools=[search_tool, response])

agent = get_agent()

def handle_query(query: str):
    # results = qa_prompt(get_llm_chat_model(), get_vector_database(), query)
    output = agent.invoke(
        {
            "role": "user",
            "content": query 
        }
    )
    response, documents = output
    return response, documents

def ui():
    chat_tab, trace_tab = st.tabs(["Chat", "Trace"])
    documents= None
    query: str = st.chat_input("Interact with the AI assistant...")
    
    with chat_tab:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        else:
            for message in st.session_state.messages:
                chat_tab.chat_message(message["role"]).write(message["content"], unsafe_allow_html=True)
        
        if query:
            st.session_state.messages.append({
                "role": "user",
                "content": query
            })
            chat_tab.chat_message("user").write(query)
            output = handle_query(query)
            content, documents = output
            print("Content:", content, "Documents:", documents)
            ai_message = content.get("content")
            if documents:
                docs = f"** Reference PMIDs: {', '.join([doc.metadata['file_name'].stem for doc in documents])} **"
                ai_message = ai_message + docs
            else:
                pass
            chat_tab.chat_message("bot").write(ai_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_message
            })

    with trace_tab:
        trace_tab.empty()
        for message in agent.trace:
            trace_tab.write(message)
        
if __name__ == "__main__":
    load_dotenv()
    ui()
