import streamlit as st
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import os
 
load_dotenv()  # works locally
 
# Get API key - works on both local and Streamlit Cloud
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
 
 
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Returns compact results to save tokens."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))  # 3 results instead of 5
            if not results:
                return "No search results found."
            output = ""
            for r in results:
                # Truncate each summary to 250 chars to save tokens
                summary = r['body'][:250] + "..." if len(r['body']) > 250 else r['body']
                output += f"Title: {r['title']}\nSummary: {summary}\n\n"
            return output
    except Exception as e:
        return f"Search failed: {str(e)}"
 
 
# --- Streamlit UI ---
st.title("🤖 My AI Agent")
st.caption("Powered by CrewAI + Gemini | Web Search enabled")
 
# Keep chat history across messages
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
# Chat input
if prompt := st.chat_input("Ask me anything..."):
 
    # Save and show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    # Build conversation history - only last 3 exchanges to save tokens
    recent = st.session_state.messages[-7:-1]  # last 3 user+assistant pairs
    history = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
        history += f"{role}: {content}\n"
 
    with st.chat_message("assistant"):
        with st.spinner("Searching the web and writing response..."):
 
            # Step 1: Run web search before the crew
            search_results = web_search(prompt)
 
            # Debug panel - remove when everything works
            with st.expander("🔍 Search results (debug)"):
                st.text(search_results)
 
            llm = LLM(model="gemini-2.5-flash", api_key=api_key)
 
            # Agent 1: Researcher — analyzes search results
            researcher = Agent(
                role="Research Specialist",
                goal="Analyze web search results and extract the most relevant information to answer the user's question",
                backstory="You are an expert researcher who reads search results and identifies key facts.",
                llm=llm,
                verbose=False
            )
 
            # Agent 2: Writer — turns research into a friendly response
            writer = Agent(
                role="Communication Specialist",
                goal="Write a clear, friendly response based on the research",
                backstory="You explain things simply and conversationally.",
                llm=llm,
                verbose=False
            )
 
            # Task 1: Researcher analyzes search results
            research_task = Task(
                description=f"""Web search results:
{search_results}
 
Recent conversation:
{history}
 
User's question: {prompt}
 
Extract the key facts from the search results that answer this question.""",
                expected_output="A concise summary of the key facts from the search results.",
                agent=researcher
            )
 
            # Task 2: Writer crafts the final response
            write_task = Task(
                description=f"""Using the research, write a friendly response to: {prompt}
 
Keep it concise and conversational. Use the conversation history for context:
{history}""",
                expected_output="A clear, friendly answer based on the research.",
                agent=writer,
                context=[research_task]
            )
 
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                verbose=False
            )
 
            try:
                result = crew.kickoff()
                response = str(result)
                st.write(response)
            except Exception as e:
                st.error(f"Error type: {type(e).__name__}")
                st.error(f"Error details: {str(e)}")
                response = "Error occurred."
 
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})