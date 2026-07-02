mport streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import os
 
load_dotenv()  # works locally
 
# Get API key - works on both local and Streamlit Cloud
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
 
# --- Web Search Tool (free, no API key needed) ---
class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for current, real-time information on any topic. Use this for recent events, facts, or anything needing up-to-date information."
 
    def _run(self, query: str) -> str:
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
 
    # Run the two-agent crew
    with st.chat_message("assistant"):
        with st.spinner("Researching and writing..."):
 
            llm = LLM(model="gemini-2.5-flash", api_key=api_key)
            search_tool = WebSearchTool()
 
            # Agent 1: Researcher with web search
            researcher = Agent(
                role="Research Specialist",
                goal="Search the web and gather accurate, up-to-date information to answer the user's question",
                backstory="You are an expert researcher who finds reliable information using web search. You always look for the most current information available.",
                llm=llm,
                tools=[search_tool],
                verbose=False
            )
 
            # Agent 2: Writer
            writer = Agent(
                role="Communication Specialist",
                goal="Take research findings and write a clear, friendly, conversational response",
                backstory="You are a skilled writer who takes complex research and explains it simply. You write in a warm, conversational tone suitable for a beginner.",
                llm=llm,
                verbose=False
            )
 
            # Task 1: Research the question
            research_task = Task(
                description=f"""Previous conversation:
{history}
 
User's current question: {prompt}
 
Search the web to find accurate and up-to-date information to answer this question. Gather key facts and relevant details.""",
                expected_output="A detailed summary of research findings with key facts and sources.",
                agent=researcher
            )
 
            # Task 2: Write a response based on the research
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