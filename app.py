import streamlit as st
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

load_dotenv()

st.title("🤖 My AI Agent")
st.caption("Powered by CrewAI + Gemini")

# Keep chat history across messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box at the bottom
if prompt := st.chat_input("Ask me anything..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Run the agent and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = LLM(
                model="gemini-2.5-flash",
                api_key=os.getenv("GEMINI_API_KEY")
            )
            researcher = Agent(
                role="Research Assistant",
                goal="Answer questions clearly and helpfully",
                backstory="You are a friendly expert who explains things simply.",
                llm=llm,
                verbose=False
            )
            task = Task(
                description=prompt,
                expected_output="A clear, helpful answer to the user's question.",
                agent=researcher
            )
            crew = Crew(agents=[researcher], tasks=[task], verbose=False)
            result = crew.kickoff()
            response = str(result)
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})