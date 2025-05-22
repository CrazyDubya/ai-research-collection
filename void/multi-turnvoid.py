
import os
import json
import uuid
import random
import re
import requests
import time
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
    def __init__(self, name, archetype, agency):
        self.name = name
        self.archetype = archetype
        self.agency = agency
        self.meta_memory = []
        self.emotional_state = "neutral"
        self.internal_thoughts = []

    def observe_scene(self, all_thoughts, cue_text):
        recent_other_thoughts = [
            f"{name}: {thought.splitlines()[0]}"
            for name, thought in all_thoughts.items()
            if name != self.name
        ]
        combined_observation = "\\n".join(recent_other_thoughts[-3:])
        return f"CUE: {cue_text or 'None'}\\nOTHERS:\\n{combined_observation or 'None'}"

    def think(self, cue_text, all_thoughts):
        observation = self.observe_scene(all_thoughts, cue_text)
        prompt = f"""
You are {self.name}, a {self.archetype} actor in a shared space with others.
Your current emotion is '{self.emotional_state}'.
Your memory includes: {", ".join(self.meta_memory[-3:]) or 'Nothing'}.

{observation}

Respond with:
THOUGHT: (your inner reflection)
PERCEPTION: (what you notice)
ACTION: (what you do or intend)
EMOTION: (how your emotion changes)
"""
        try:
            response = run_grok_prompt(user_prompt=prompt)
        except Exception as e:
            response = f"[ERROR]: {e}"

        self.internal_thoughts.append(response)
        self.meta_memory.append(response.splitlines()[0])
        print(f"\\n[{self.name}]\\n{response}\\n")
        return response

# === Void Engine ===
def simulate_ticks(actors, cue_sequence, ticks=5):
    all_thoughts = {}
    for tick in range(ticks):
        print(f"\\n====== TICK {tick + 1} ======")
        cue = cue_sequence[tick] if tick < len(cue_sequence) else ""
        for actor in actors:
            thought = actor.think(cue, all_thoughts)
            all_thoughts[actor.name] = thought
        time.sleep(1)

# === Entry Point ===
def main():
    actors = [
        Actor("Orra", "Archivist", 42),
        Actor("Cellen", "Disbeliever", 87),
        Actor("Jun", "Trickster", 69),
    ]
    cue_sequence = [
        "A low-frequency hum begins beneath the floorboards.",
        "The lights flicker for a moment, then stabilize.",
        "You hear the faint sound of a coin rolling.",
        "The temperature drops by a few degrees.",
        "Something unseen brushes past your shoulder."
    ]
    simulate_ticks(actors, cue_sequence, ticks=5)

if __name__ == "__main__":
    main()
