from backend.vector_store import VectorStore
from backend.document_loader import load_policy_chunks
from backend.orders_tool import extract_order_id, get_order_by_id
from backend.openrouter_client import generate_answer

_policy_store = VectorStore(name="policy_index")


def init_support_agent():
    """Call once at startup to build/load the policy vector index."""
    chunks = load_policy_chunks()
    _policy_store.build_or_load(chunks)


SUPPORT_SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer the user's question using ONLY the policy context provided below.
If the context doesn't contain the answer, say you don't have that
information rather than guessing. Be concise and friendly.

Context:
{context}
"""

ORDER_SYSTEM_PROMPT = """You are a helpful customer support assistant.
The user is asking about an order. Here is the order's data:

{order_data}

Answer their question using this data. If the order was not found,
politely say so and ask them to double check the order ID.
"""


def _answer_order_question(user_message: str):
    order_id = extract_order_id(user_message)
    if not order_id:
        return None  # not actually an order-id lookup; fall through to policy RAG

    order = get_order_by_id(order_id)
    if order is None:
        prompt = ORDER_SYSTEM_PROMPT.format(order_data=f"No order found with ID '{order_id}'.")
    else:
        prompt = ORDER_SYSTEM_PROMPT.format(order_data=order)
    return generate_answer(prompt, user_message)


def _answer_policy_question(user_message: str) -> str:
    relevant_chunks = _policy_store.search(user_message, top_k=4)
    context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c in relevant_chunks)
    prompt = SUPPORT_SYSTEM_PROMPT.format(context=context or "No relevant context found.")
    return generate_answer(prompt, user_message)


def answer(user_message: str) -> str:
    lowered = user_message.lower()
    looks_like_order_question = "order" in lowered or extract_order_id(user_message) is not None

    if looks_like_order_question:
        order_answer = _answer_order_question(user_message)
        if order_answer is not None:
            return order_answer

    return _answer_policy_question(user_message)