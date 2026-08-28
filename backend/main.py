from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from backend.routing_agent import route_query
from backend import support_agent
from backend import product_agent
from backend.openrouter_client import generate_answer

app = FastAPI(title="Two-Agent RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    print("Building/loading vector indexes...")
    support_agent.init_support_agent()
    product_agent.init_product_agent()
    print("Ready.")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    route: str
    answer: str


MERGE_PROMPT = """You are combining two partial answers into one coherent
reply to the user's original question. Do not repeat information; blend
them naturally.

Support agent's answer:
{support_answer}

Product agent's answer:
{product_answer}
"""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    route = route_query(req.message)

    if route == "support":
        answer = support_agent.answer(req.message)
    elif route == "product":
        answer = product_agent.answer(req.message)
    else:  # ambiguous -> ask both agents, then merge
        support_answer = support_agent.answer(req.message)
        product_answer = product_agent.answer(req.message)
        merge_prompt = MERGE_PROMPT.format(
            support_answer=support_answer, product_answer=product_answer
        )
        answer = generate_answer(merge_prompt, req.message)

    return ChatResponse(route=route, answer=answer)


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))