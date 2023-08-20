import os

import streamlit as st
import weaviate
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.vectorstores import Weaviate

from src.schema import DOC_CLASS

load_dotenv()

st.set_page_config(page_title="TAix", page_icon="💸")
st.title("💸 TAix - Tax Advice Agent")


@st.cache_resource(ttl="1h")
def configure_retriever_wv():
    client = weaviate.Client(
        "http://localhost:8080", additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]}
    )
    vectorstore = Weaviate(client, DOC_CLASS, "country")
    return vectorstore.as_retriever(search_kwargs={"k": 20})


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


def chat_with_doc():
    if "qa_chain" not in st.session_state:
        vector_store = configure_retriever_wv()

        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store, memory=memory, verbose=True)

        st.session_state["qa_chain"] = qa_chain
        st.session_state["retriever"] = vector_store

    qa_chain = st.session_state.qa_chain

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask me anything!")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(question=user_query, callbacks=[retrieval_handler, stream_handler])
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat_with_doc()
