import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.tools import BaseTool

from src.taix.client import wv_retriever

load_dotenv()

st.set_page_config(page_title="TAix", page_icon="💸")
st.title("💸 TAix - Tax Advice Agent")

WV_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")


class TaxiInvoiceTool(BaseTool):
    name = "invoice_tool"
    description = "useful for when you need to get information about an invoices"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""

        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        vector_store = wv_retriever(WV_URL)

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store, verbose=True, memory=memory)

        return qa_chain.run(question=query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


def chat_with_doc():
    if "agent" not in st.session_state:
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
        tools = load_tools(["llm-math"], llm=llm)
        tools.extend([TaxiInvoiceTool()])
        # tools = [TaxiInvoiceTool()]
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
        )
        st.session_state["agent"] = agent

    agent = st.session_state.agent

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(prompt, callbacks=[st_callback])
            st.write(response)


if __name__ == "__main__":
    chat_with_doc()
