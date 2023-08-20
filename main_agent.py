import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory

load_dotenv()

st.set_page_config(page_title="TAix", page_icon="💸")
st.title("💸 TAix - Tax Advice Agent")


def chat_with_doc():
    if "agent" not in st.session_state:
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
        tools = load_tools(["ddg-search"])
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
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
