from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

# Load your API key from the .env file
load_dotenv()

# Connect to Gemini (free model)
llm = LLM(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)
# Create an AI agent with a role and personality
researcher = Agent(
    role="Research Assistant",
    goal="Answer questions clearly and helpfully",
    backstory="You are a friendly expert who explains things simply for beginners.",
    llm=llm,
    verbose=True  # Shows you what the agent is thinking
)

# Give the agent a task
task = Task(
    description="Explain what generative AI is in 3 simple sentences. Use plain English.",
    expected_output="A clear 3-sentence explanation of generative AI for a complete beginner.",
    agent=researcher
)

# Run everything
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()

print("\n========== AGENT RESULT ==========")
print(result)