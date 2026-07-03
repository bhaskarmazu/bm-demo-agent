import streamlit as st
from crewai import Agent, Task, Crew, LLM
from tavily import TavilyClient
from dotenv import load_dotenv
import os
 
load_dotenv()  # works locally
 
# Get API keys - works on both local and Streamlit Cloud
gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
tavily_key = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))
 
 
def web_search(query: str) -> str:
    """Search the web using Tavily (reliable from cloud servers)."""
    try:
        client = TavilyClient(api_key=tavily_key)
        response = client.search(query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "No search results found."
        output = ""
        for r in results:
            summary = r.get("content", "")[:250]
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
 
    # Build conversation history - last 3 exchanges to save tokens
    recent = st.session_state.messages[-7:-1]
    history = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        history += f"{role}: {content}\n"
 
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
 
            # Build a context-aware search query for follow-up questions
            search_query = prompt
            if history and len(prompt.split()) < 8:
                # Short question = likely a follow-up, enrich with recent context
                last_lines = [l for l in history.strip().split('\n') if l.strip()]
                if last_lines:
                    context_hint = last_lines[-1].replace("Assistant: ", "").replace("User: ", "")[:100]
                    search_query = f"{context_hint} {prompt}"
 
            # Step 1: Web search
            search_results = web_search(search_query)
 
            # Debug panel - remove when no longer needed
            with st.expander("🔍 Search results (debug)"):
                st.text(f"Query used: {search_query}\n\n{search_results}")
 
            # Step 2: Run agent
            try:
                llm = LLM(model="gemini-2.5-flash", api_key=gemini_key)
 
                agent = Agent(
                    role="Research Assistant",
                    goal="Find information and give clear, friendly answers",
                    backstory="You are a helpful expert who reads search results and explains things simply and conversationally.",
                    llm=llm,
                    verbose=False
                )
 
                task = Task(
                    description=f"""Web search results:
{search_results}
 
Conversation history (IMPORTANT - use this for context on follow-up questions):
{history}
 
User's current question: {prompt}
 
Use the conversation history to understand what the user is referring to (e.g. if they say "who won it?", check the history to know what "it" refers to).
Then use the search results to give an accurate, friendly answer.
If the search results don't contain the answer, say so honestly.""",
                    expected_output="A concise, friendly answer that correctly handles follow-up questions using conversation context.",
                    agent=agent
                )
 
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                result = crew.kickoff()
                response = str(result)
                st.write(response)
 
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {str(e)}")
                response = "Sorry, I encountered an error. Please try again."
                st.write(response)
 
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})