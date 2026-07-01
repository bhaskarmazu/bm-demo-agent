
App · PY
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os
 
load_dotenv()  # works locally
 
# Get API key - works on both local and Streamlit Cloud
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
 
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
 
    # Save and show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    # Build conversation history from all previous messages
    history = ""
    for msg in st.session_state.messages[:-1]:  # exclude current message
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
 
    # Run the agent and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = LLM(
                model="gemini-2.5-flash",
                api_key=api_key
            )
            researcher = Agent(
                role="Research Assistant",
                goal="Answer questions clearly and helpfully",
                backstory="You are a friendly expert who explains things simply.",
                llm=llm,
                verbose=False
            )
            task = Task(
                description=f"""Here is the conversation so far:
{history}
Now respond to this new message: {prompt}
 
Take the conversation history into account in your response.""",
                expected_output="A helpful answer that considers the full conversation context.",
                agent=researcher
            )
            crew = Crew(agents=[researcher], tasks=[task], verbose=False)
            result = crew.kickoff()
            response = str(result)
            st.write(response)
 
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
 