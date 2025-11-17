

# imports
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from llmTools import get_weather
import os

# constants
app = FastAPI()
KEY_CHATGPT = os.environ.get('MARC_CHAT_KEY')

# classes
class PromptRequest(BaseModel):
    prompt: str

class ResponseModel(BaseModel):
    response: str

# Initialize LLM
llm = OpenAI(openai_api_key=KEY_CHATGPT, temperature=0.9)
# llm = OpenAI(temperature=0.7)

# Agent with custom tools
tools = [get_weather]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@app.post("/ask", response_model=ResponseModel)
async def ask_agent(req: PromptRequest):
    response = agent.run(req.prompt)
    return ResponseModel(response=response)


# to run
# uvicorn main:app --reload