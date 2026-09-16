"""
Chat-model call for explaining a modified chunk pair.

Requires:
    ollama pull llama3.1     (or swap CHAT_MODEL for whatever you have pulled)
"""

import requests

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.1"


def explain_change(old_text: str, new_text: str) -> str:
    prompt = f"""You are comparing two versions of a document section. In 1-2 concise \
sentences, explain WHAT changed and why it might matter. Do not restate both full texts.

VERSION A:
{old_text}

VERSION B:
{new_text}

Explanation:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()
