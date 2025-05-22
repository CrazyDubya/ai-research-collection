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


def run_grok_prompt(prompt, max_tokens=None):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "grok-3-beta",
        "deferred": True
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

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
            "Lighting Director": f"""You are {self.name}, CREW member working as Lighting Director for this theatrical production.
BRIEF RESPONSE REQUIRED (50 words max).
Create 2-3 concise lighting states that evoke emotion WITHOUT technical jargon.
NO character dialogue or actions - you are CREW, not an actor.
Context: {context}
Summary of last tick: {summary}
LIGHTING:""",

            "Cue Team Lead": f"""You are {self.name}, CREW member working as Cue Provider for this theatrical production.
BRIEF RESPONSE REQUIRED (75 words max).
Provide 1-2 essential cues that drive the scene forward.
NO elaborate backstory or character development - you are CREW, not an actor.
Context: {context}
Summary of last tick: {summary}
CUE:""",

            "Prop Master": f"""You are {self.name}, CREW member working as Prop Master for this theatrical production.
BRIEF RESPONSE REQUIRED (50 words max).
Describe how actors interact with 1-2 key props only.
NO elaborate prop descriptions or history - you are CREW, not an actor.
Context: {context}
Summary of last tick: {summary}
PROPS:"""
        }

        prompt = role_prompts[self.role]
        max_tokens = {"Lighting Director": 150, "Cue Team Lead": 200, "Prop Master": 150}[self.role]
        raw = run_grok_prompt(prompt, max_tokens)

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
Backstory: {self.character.get('backstory', 'Develop this as you go')}
Relationships: {self.character.get('relationships', 'Develop these as you go')}

Your recent acting choices: {self.memory[-3:] if self.memory else "First scene"}
Cue: {cue_text}
Summary of last tick: {summary}

Create a response that:
1. Shows clear character motivations beyond simple reactions
2. Adds dimension to your character beyond their primary trait
3. Creates specific physical actions actors can perform
4. EMPHASIZES MEANINGFUL DIALOGUE (IMPORTANT: your character must speak several lines)
5. Develops your character's arc within the scene
6. INTERACTS directly with at least one other character

Respond with:
THOUGHT: (1-2 sentences on genuine motivation, not technical details)
PERCEPTION: (What your character notices that others might miss)
ACTION: (Specific, varied, playable physical action)
SPEAK: (REQUIRED - Multiple lines of authentic dialogue that reveals character)
EMOTION: (Nuanced feeling beyond the obvious)"""

        # Limit token count to encourage more focused responses
        result = run_grok_prompt(prompt, max_tokens=400)
        self.memory.append(result.splitlines()[0])
        return result


def enhance_character_interactions(actor_responses, character_names):
    """Post-process actor responses to encourage more interaction."""
    if not actor_responses:
        return {}

    characters_present = ", ".join(character_names)

    interaction_prompt = f"""
    Characters present in this scene: {characters_present}

    The following are character responses in the current scene:
    {json.dumps(actor_responses, indent=2)}

    For EACH character, suggest ONE specific line of dialogue that directly addresses another character by name.
    Make sure these dialogue lines:
    1. Are authentic to each character's personality
    2. Move the scene forward
    3. Create tension or connection between characters
    4. Are brief and impactful (1-2 sentences max)

    Format as:
    CHARACTER_NAME: "Dialogue line directed at another character."
    """

    interaction_suggestions = run_grok_prompt(interaction_prompt, max_tokens=200)
    return interaction_suggestions


def summarize_tick(actors_responses, crew_responses):
    summary_prompt = f"""Summarize the scene concisely:
1. Focus on meaningful character interactions, not technical elements
2. Highlight clear cause-and-effect relationships between actions
3. Note evolving character relationships and emotions
4. Mention only the most important technical elements that affect story
5. STRICT LIMIT: 75 words maximum

Actors: {actors_responses}
Crew: {crew_responses}"""
    return run_grok_prompt(summary_prompt, max_tokens=150)


# Expanded character definitions with relationships and backstory
characters = [
    {
        "name": "Orra", "gender": "female", "role": "Archivist",
        "habits": "Meticulous, reflective",
        "backstory": "A scholar who discovered the archive accidentally while researching her family history. Has a mysterious connection to the objects within.",
        "relationships": "Distrusts Jun's motives, relies on Mara for emotional support, respects Cellen's intellect despite disagreements."
    },
    {
        "name": "Jun", "gender": "male", "role": "Trickster",
        "habits": "Playful, mischievous",
        "backstory": "Knows more about the archive than he reveals. Has visited before and may have unleashed its power deliberately.",
        "relationships": "Enjoys provoking Cellen, fascinated by Orra's connection to the archive, wary of Mara's perceptiveness."
    },
    {
        "name": "Cellen", "gender": "male", "role": "Disbeliever",
        "habits": "Skeptical, rational",
        "backstory": "A journalist investigating strange occurrences connected to the archive. Has personal reasons to deny supernatural explanations.",
        "relationships": "Frustrated by Jun's games, protective of Orra despite his skepticism, respectful of Mara's practical approach."
    },
    {
        "name": "Mara", "gender": "female", "role": "Caretaker",
        "habits": "Caring, observant",
        "backstory": "Descended from a long line of guardians who have watched over the archive. Knows its dangers but not all its secrets.",
        "relationships": "Suspects Jun knows more than he says, worried about Orra's obsession with the archive, appreciates Cellen's grounding influence."
    },
    {"name": "Lena", "gender": "female", "role": "Visionary", "habits": "Dreamy, intuitive"},
    {"name": "Hale", "gender": "male", "role": "Guardian", "habits": "Protective, alert"}
]

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

assigned = set()
for actor in actors:
    for character in characters:
        if character['gender'] == actor.gender and character['name'] not in assigned:
            actor.inhabit(character)
            assigned.add(character['name'])
            break

# Enhanced cue sequence with more narrative progression
cue_sequence = [
    "A low-frequency hum vibrates beneath the floorboards as the archive door shuts behind you.",
    "The lights flicker as ancient dust swirls around a forgotten journal on the central desk.",
    "A coin rolls across the floor, stopping at the edge of a hidden trapdoor.",
    "The journal pages begin to turn on their own, revealing symbols that glow in the dim light.",
    "The temperature drops suddenly, causing breath to become visible in the air.",
    "A whisper seems to come from inside the walls, calling someone's name.",
    "The artifact cabinet rattles violently then falls completely silent.",
    "The symbols from the journal begin appearing faintly on the skin of whoever touched it last."
]


def simulate_ticks(actors, crew, cue_sequence, ticks=3):
    summary = ""
    tick_records = []

    for tick in range(ticks):
        print(f"\n====== TICK {tick + 1} ======")
        # Cycle through cues or repeat the last one if we run out
        cue_index = min(tick, len(cue_sequence) - 1)
        cue = cue_sequence[cue_index]

        crew_responses = {}
        for member in crew:
            directives = member.act(cue, summary)
            print(f"[{member.name} - {member.role}]\n{json.dumps(directives, indent=2)}\n")
            crew_responses[member.name] = directives
            time.sleep(1)

        actors_responses = {}
        character_names = []
        for actor in actors:
            if actor.character:
                character_names.append(actor.character['name'])
            result = actor.think(
                cue_text=crew_responses.get("Tarn Vale", {}).get("cue_text_override", cue),
                summary=summary
            )
            print(f"[{actor.actor_name} as {actor.character['name']} - {actor.character['role']}]\n{result.strip()}\n")
            actors_responses[actor.actor_name] = result
            time.sleep(1)

        # Enhance interactions
        if tick > 0:  # Skip first tick to establish characters
            interaction_suggestions = enhance_character_interactions(actors_responses, character_names)
            print(f"[INTERACTION SUGGESTIONS]\n{interaction_suggestions.strip()}\n")

        summary = summarize_tick(actors_responses, crew_responses)
        print(f"[TICK {tick + 1} SUMMARY]\n{summary.strip()}\n")

        # Save tick data
        tick_data = {
            "tick": tick + 1,
            "cue": cue,
            "crew_responses": crew_responses,
            "actors_responses": actors_responses,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        tick_records.append(tick_data)

        # Periodically save simulation state
        if tick % 5 == 0 or tick == ticks - 1:
            with open(SAVE_FILE, "w") as f:
                json.dump(tick_records, f, indent=2)


simulate_ticks(actors, crew, cue_sequence, ticks=30)