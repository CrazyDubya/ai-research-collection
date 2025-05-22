import os
import json
import uuid
import requests
import time
from datetime import datetime
from tenacity import retry, wait_exponential

SAVE_FILE = "void_simulation_state.json"

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_DEFERRED_URL = "https://api.x.ai/v1/chat/deferred-completion"
GROK_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer xai-z4RK1RiQpcxbrprxdshRSKdykgyg4CGEHDKtwp1FqdAKp0lfSsnXbRFlgJieIHRjDwasEbqMqZz9wPC1"
}

@retry(wait=wait_exponential(multiplier=1, min=1, max=60))
def get_grok_completion(request_id):
    response = requests.get(f"{GROK_DEFERRED_URL}/{request_id}", headers=GROK_HEADERS)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 202:
        raise Exception("Response not ready yet")
    else:
        raise Exception(f"{response.status_code} Error: {response.text}")

def run_grok_prompt(prompt):
    payload = {"messages": [{"role": "user", "content": prompt}], "model": "grok-3-beta", "deferred": True}
    response = requests.post(GROK_API_URL, headers=GROK_HEADERS, json=payload)
    response.raise_for_status()
    request_id = response.json()["request_id"]
    completion = get_grok_completion(request_id)
    return completion['choices'][0]['message']['content']

class CrewMember:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.memory = []

    def act(self, context, summary):
        role_prompts = {
            "Lighting Director": f"""You are {self.name}, Lighting Director. Precisely adjust lighting with exact details: color, intensity (%), direction, focus, and motion.
Context: {context}
Summary of last tick: {summary}
LIGHTING:""",

            "Cue Team Lead": f"""You are {self.name}, Cue Provider and NOT part of the show. Provide exact dialogue or actionable stage direction.
Context: {context}
Summary of last tick: {summary}
CUE:""",

            "Prop Master": f"""You are {self.name}, Prop Master. Clearly specify props with detailed description and exact placement.
Context: {context}
Summary of last tick: {summary}
PROPS:"""
        }

        prompt = role_prompts[self.role]
        raw = run_grok_prompt(prompt)

        directives = {}
        if self.role == "Lighting Director":
            directives["lighting_changes"] = raw.strip()
        elif self.role == "Cue Team Lead":
            directives["cue_text_override"] = raw.strip()
        elif self.role == "Prop Master":
            directives["props_to_add"] = raw.strip()

        return directives

class Actor:
    def __init__(self, actor_name, gender):
        self.actor_name = actor_name
        self.gender = gender
        self.memory = []
        self.character = None

    def inhabit(self, character):
        self.character = character

    def think(self, cue_text, summary):
        if not self.character:
            return f"[ERROR] {self.actor_name} is not inhabiting any character."

        prompt = f"""
You are {self.actor_name}, portraying {self.character['name']} ({self.character['role']}).
Character habits: {self.character['habits']}
Your actor habits: {self.memory[-3:]}
Cue: {cue_text}
Summary of last tick: {summary}
Respond with:
THOUGHT:
PERCEPTION:
ACTION:
SPEAK: (optional)
EMOTION:"""
        result = run_grok_prompt(prompt)
        self.memory.append(result.splitlines()[0])
        return result

def summarize_tick(actors_responses, crew_responses):
    summary_prompt = f"Summarize concisely and densely:\nActors: {actors_responses}\nCrew: {crew_responses}"
    return run_grok_prompt(summary_prompt)

crew = [
    CrewMember("Salma Rayne", "Lighting Director"),
    CrewMember("Tarn Vale", "Cue Team Lead"),
    CrewMember("Greel", "Prop Master")
]

actors = [
    Actor("Elias", "male"),
    Actor("Marcus", "male"),
    Actor("Anna", "female"),
    Actor("Elise", "female")
]

characters = [
    {"name": "Orra", "gender": "female", "role": "Archivist", "habits": "Meticulous, reflective"},
    {"name": "Jun", "gender": "male", "role": "Trickster", "habits": "Playful, mischievous"},
    {"name": "Cellen", "gender": "male", "role": "Disbeliever", "habits": "Skeptical, rational"},
    {"name": "Mara", "gender": "female", "role": "Caretaker", "habits": "Caring, observant"},
    {"name": "Lena", "gender": "female", "role": "Visionary", "habits": "Dreamy, intuitive"},
    {"name": "Hale", "gender": "male", "role": "Guardian", "habits": "Protective, alert"}
]

assigned = set()
for actor in actors:
    for character in characters:
        if character['gender'] == actor.gender and character['name'] not in assigned:
            actor.inhabit(character)
            assigned.add(character['name'])
            break

cue_sequence = [
    "A low-frequency hum vibrates beneath the floorboards.",
    "The lights flicker briefly, then stabilize.",
    "You hear a coin rolling into darkness."
]

def simulate_ticks(actors, crew, cue_sequence, ticks=3):
    summary = ""
    for tick in range(ticks):
        print(f"\n====== TICK {tick + 1} ======")
        cue = cue_sequence[tick] if tick < len(cue_sequence) else ""

        crew_responses = {}
        for member in crew:
            directives = member.act(cue, summary)
            print(f"[{member.name} - {member.role}]\n{json.dumps(directives, indent=2)}\n")
            crew_responses[member.name] = directives
            time.sleep(1)

        actors_responses = {}
        for actor in actors:
            result = actor.think(
                cue_text=crew_responses.get("Tarn Vale", {}).get("cue_text_override", cue),
                summary=summary
            )
            print(f"[{actor.actor_name} as {actor.character['name']} - {actor.character['role']}]\n{result.strip()}\n")
            actors_responses[actor.actor_name] = result
            time.sleep(1)

        summary = summarize_tick(actors_responses, crew_responses)
        print(f"[TICK {tick + 1} SUMMARY]\n{summary.strip()}\n")

simulate_ticks(actors, crew, cue_sequence, ticks=30)
