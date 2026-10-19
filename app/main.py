
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ddgs import DDGS
from groq import Groq

load_dotenv()

app = FastAPI(title="Google Search AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


@app.get("/")
def home():
    return {"message": "Google Search AI Agent is running!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/search")
def search_web(query: str):
    try:
        results = DDGS().text(query, max_results=5)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/ask")
def ask_agent(query: str):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing from .env"
        )

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No search results found."
            )

        search_text = "\n".join(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('href', '')}\n"
            f"Summary: {item.get('body', '')}"
            for item in results
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using the supplied "
                        "web search results. Be clear and beginner-friendly. "
                        "If the results do not support an answer, say so. "
                        "Do not invent facts or citations."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Web search results:\n{search_text}"
                    )
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", "")
                }
                for item in results
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
