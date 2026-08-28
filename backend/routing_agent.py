from backend.openrouter_client import generate_answer

ROUTING_PROMPT = """You are a query router for a customer support and product chatbot.
Classify the user's message into exactly one category:

- "support": questions about company info, policies (returns, shipping, warranty, payment), or an existing order's status/details.
- "product": questions about products, their specs, prices, availability, or recommendations.
- "ambiguous": the message clearly touches both a product AND an order/policy in a way that needs both agents (e.g. "can I return this product if it's still under warranty").

Respond with ONLY one word: support, product, or ambiguous. No punctuation, no explanation.
"""


def route_query(user_message: str) -> str:
    result = generate_answer(ROUTING_PROMPT, user_message).strip().lower()
    if "support" in result:
        return "support"
    if "product" in result:
        return "product"
    if "ambig" in result:
        return "ambiguous"
    return "support"