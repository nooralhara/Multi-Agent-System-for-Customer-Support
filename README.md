# Multi-Agent-System-for-Customer-Support


## TechStore Multi-Agent Support Chatbot

A two-agent Retrieval-Augmented Generation (RAG) chatbot for a fictional electronics store. A routing agent classifies each incoming message and dispatches it to a Support Agent (company policies + order lookups) or a Product Agent (product catalog search), then returns a single answer through a FastAPI backend and a HTML/Tailwind frontend.

*How it works* 


The frontend posts the user's message to POST /chat.

routing_agent.py asks an LLM to classify the message as support, product, or ambiguous.
Depending on the route:
support → support_agent.py either looks up an order (via orders_tool.py against data/orders.csv) or answers from the five policy PDFs in data/, retrieved through a Chroma vector index.
product → product_agent.py answers from data/products.csv, also via a Chroma vector index.
ambiguous → both agents answer independently, then a third LLM call merges the two answers into one response.

On server startup, main.py builds (or loads, if already built) both Chroma collections — one for policy chunks, one for product chunks — under vector_index/ (Used in local hosting).



## Tech stack
Backend: FastAPI + Uvicorn

LLM access: OpenAI SDK pointed at OpenRouter's OpenAI-compatible API

Generation model: openrouter/free (auto-routes to the least-busy free model)

Embedding model: liquid/lfm-2.5-embedding-350m:free

Vector store: ChromaDB (PersistentClient, local on-disk storage)

Data parsing: pypdf for the policy PDFs, pandas for the order/product CSVs

Frontend: HTML + Tailwind CSS + JS




## Project structure

backend/
  config.py            # loads .env
  openrouter_client.py # all OpenRouter API calls
  document_loader.py   # extracts + chunks the policy PDFs
  orders_tool.py        # order ID extraction + CSV lookup, no LLM involved
  vector_store.py       # Chroma-backed VectorStore class
  routing_agent.py      # classifies each message: support / product / ambiguous
  support_agent.py      # order lookups + policy RAG
  product_agent.py      # product catalog RAG
  main.py                # FastAPI app, startup hook, /chat endpoint, static serving
  
data/
  company_info.pdf, payment_policy.pdf, return_policy.pdf,
  shipping_policy.pdf, warranty_policy.pdf   # policy knowledge base
  orders.csv            # sample order records
  products.csv          # product catalog
  
frontend/
  index.html             # chat UI
requirements.txt         # Python Dependencies


## Notes:

This application has only been tested and hosted locally.

Created by NoorAldeen Al-Hara
