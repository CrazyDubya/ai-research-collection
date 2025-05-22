
import os
import json
import uuid
import random
import re
import requests
from datetime import datetime
from collections import deque, defaultdict
from tenacity import retry, wait_exponential

# === GROK LLM Integration ===
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_DEFERRED_URL = "https://api.x.ai/v1/chat/deferred-completion"
GROK_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}

def start_grok_chat(messages, model="grok-3-beta", system_prompt=None):
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "messages": full_messages,
        "model": model,
        "deferred": True
    }

    response = requests.post(GROK_API_URL, headers=GROK_HEADERS, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["request_id"]

@retry(wait=wait_exponential(multiplier=1, min=1, max=60))
def get_grok_completion(request_id):
    response = requests.get(f"{GROK_DEFERRED_URL}/{request_id}", headers=GROK_HEADERS)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 202:
        raise Exception("Response not ready yet")
    else:
        raise Exception(f"{response.status_code} Error: {response.text}")

def run_grok_prompt(user_prompt, system_prompt="You are a creative and observant persona.", model="grok-3-beta"):
    request_id = start_grok_chat(
        messages=[{"role": "user", "content": user_prompt}],
        model=model,
        system_prompt=system_prompt
    )
    print(f"Request ID: {request_id}")
    completion = get_grok_completion(request_id)
    return completion['choices'][0]['message']['content']

# === Actor Simulation ===
class Actor:
    def __init__(self, name, archetype, agency, memory=None):
        self.name = name
        self.archetype = archetype
        self.agency = agency
        self.meta_memory = memory or []
        self.emotional_state = "neutral"
        self.internal_thoughts = []

    def think(self, cue_text=""):
        prompt = f"""
You are {self.name}, a {self.archetype} actor inside an experimental simulation.
Your current emotional state is '{self.emotional_state}'.
Recent memory includes: {", ".join(self.meta_memory[-3:]) or 'Nothing'}.
Active cue: {cue_text or 'none'}.

Respond in this format:
THOUGHT: (your internal reflection)
PERCEPTION: (what you notice)
ACTION: (what you attempt or prepare)
EMOTION: (how your emotion shifts or deepens)
"""
        try:
            response = run_grok_prompt(user_prompt=prompt)
        except Exception as e:
            response = f"[ERROR]: {e}"

        self.internal_thoughts.append(response)
        print(f"\n[{self.name}]\n{response}\n")

# === Void Engine ===
def simulate_void_tick(actors, cue=None):
    print("\n===== VOID TICK BEGIN =====")
    for actor in actors:
        actor.think(cue)
    print("===== VOID TICK END =====\n")

# === Entry Point ===
def main():
    actors = [
        Actor("Orra", "Archivist", 42),
        Actor("Cellen", "Disbeliever", 87),
        Actor("Jun", "Trickster", 69),
    ]
    cue = "A low-frequency hum begins to pulse beneath the floorboards."
    simulate_void_tick(actors, cue)

if __name__ == "__main__":
    main()
