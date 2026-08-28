from openai import OpenAI
from backend.config import OPENROUTER_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in result.data])
    return all_embeddings


def generate_answer(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content