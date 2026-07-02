import streamlit as st
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import os
 
load_dotenv()  # works locally
 
# Get API key - works on both local and Streamlit Cloud
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
 
 
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return results as text."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No search results found."
            output = ""
            for r in results:
                output += f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n\n"
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
 
    # Build conversation history
    history = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
 
    with st.chat_message("assistant"):
        with st.spinner("Searching the web and writing response..."):
 
            # Step 1: Run web search before the crew
            search_results = web_search(prompt)

            # Temporary debug - remove later
            with st.expander("🔍 Search results (debug)"):
                st.text(search_results)
 
            llm = LLM(model="gemini-2.5-flash", api_key=api_key)
 
            # Agent 1: Researcher — analyzes the search results
            researcher = Agent(
                role="Research Specialist",
                goal="Analyze web search results and extract the most accurate, relevant information to answer the user's question",
                backstory="You are an expert researcher who reads web search results and identifies the key facts and insights needed to answer a question.",
                llm=llm,
                verbose=False
            )
 
            # Agent 2: Writer — turns research into a friendly response
            writer = Agent(
                role="Communication Specialist",
                goal="Take research findings and write a clear, friendly, conversational response",
                backstory="You are a skilled writer who takes complex research and explains it simply. You write in a warm, conversational tone.",
                llm=llm,
                verbose=False
            )
 
            # Task 1: Researcher analyzes search results
            research_task = Task(
                description=f"""Here are the web search results for the user's question:
 
{search_results}
 
Previous conversation:
{history}
 
User's question: {prompt}
 
Analyze these search results and extract the key facts and relevant information needed to answer the question accurately.""",
                expected_output="A clear summary of the most relevant facts and information found in the search results.",
                agent=researcher
            )
 
            # Task 2: Writer crafts the final response
            write_task = Task(
                description=f"""Using the research provided, write a clear and friendly response to: {prompt}
 
Previous conversation for context:
{history}
 
Write a conversational answer that directly addresses the question. Keep it concise but complete.""",
                expected_output="A clear, friendly, conversational answer based on the research.",
                agent=writer,
                context=[research_task]  # writer receives researcher's output
            )
 
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                verbose=False
            )
 
            result = crew.kickoff()
            response = str(result)
            st.write(response)
 
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})