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

    def act(self, previous_cue, summary, tick_number):
        role_prompts = {
            "Lighting Director": f"""You are {self.name}, CREW member working as Lighting Director for this theatrical production.
BRIEF RESPONSE REQUIRED (50 words max).
Create 2-3 concise lighting states that evoke emotion WITHOUT technical jargon.
NO character dialogue or actions - you are CREW, not an actor.
Previous cue: {previous_cue}
Current tick: {tick_number}
Summary of last tick: {summary}
LIGHTING:""",

            "Cue Team Lead": f"""You are {self.name}, CREW member working as Cue Provider for this theatrical production.
BRIEF RESPONSE REQUIRED (75 words max).
Provide 1-2 essential cues that drive the scene forward.
NO elaborate backstory or character development - you are CREW, not an actor.
Previous cue: {previous_cue}
Current tick: {tick_number}
Summary of last tick: {summary}
CUE:""",

            "Prop Master": f"""You are {self.name}, CREW member working as Prop Master for this theatrical production.
BRIEF RESPONSE REQUIRED (50 words max).
Describe how actors interact with 1-2 key props only.
NO elaborate prop descriptions or history - you are CREW, not an actor.
Previous cue: {previous_cue}
Current tick: {tick_number}
Summary of last tick: {summary}
PROPS:""",

            "Sound Designer": f"""You are {self.name}, CREW member working as Sound Designer for this theatrical production.
BRIEF RESPONSE REQUIRED (50 words max).
Describe 1-2 key sound effects or ambient sounds that enhance the scene.
NO dialogue or character actions - you are CREW, not an actor.
Previous cue: {previous_cue}
Current tick: {tick_number}
Summary of last tick: {summary}
SOUND:""",

            "Stage Manager": f"""You are {self.name}, CREW member working as Stage Manager for this theatrical production.
BRIEF RESPONSE REQUIRED (60 words max).
Provide 1-2 concise notes on blocking, timing, or technical coordination.
Focus only on the technical aspects of the scene, not character development.
Previous cue: {previous_cue}
Current tick: {tick_number}
Summary of last tick: {summary}
STAGE NOTES:""",

            "Dramaturg": f"""You are {self.name}, CREW member working as Dramaturg for this theatrical production.
BRIEF RESPONSE REQUIRED (60 words max).
Provide 1 brief note on thematic elements or narrative structure.
Focus on story arc, not individual characters - you are CREW, not an actor.
Previous cue: {previous_cue}
Current tick: {tick_number}
Summary of last tick: {summary}
DRAMATURGICAL NOTE:""",

            "Scene Writer": f"""You are {self.name}, CREW member working as Scene Writer for this theatrical production.
BRIEF RESPONSE REQUIRED (80 words max).
Create a concise, evocative description of the next scene element or plot development.
This will be used to generate the main cue for the next tick.
Focus on atmosphere and narrative progression, not character development.
Previous cue: {previous_cue}
Current tick: {tick_number}
Summary of last tick: {summary}
NEXT SCENE ELEMENT:"""
        }

        if self.role not in role_prompts:
            return {"error": f"Unknown role: {self.role}"}

        prompt = role_prompts[self.role]
        max_tokens = {
            "Lighting Director": 150,
            "Cue Team Lead": 200,
            "Prop Master": 150,
            "Sound Designer": 150,
            "Stage Manager": 180,
            "Dramaturg": 180,
            "Scene Writer": 250
        }[self.role]

        raw = run_grok_prompt(prompt, max_tokens)

        directives = {}
        if self.role == "Lighting Director":
            directives["lighting_changes"] = raw.strip()
        elif self.role == "Cue Team Lead":
            directives["cue_text_override"] = raw.strip()
        elif self.role == "Prop Master":
            directives["props_to_add"] = raw.strip()
        elif self.role == "Sound Designer":
            directives["sound_effects"] = raw.strip()
        elif self.role == "Stage Manager":
            directives["stage_notes"] = raw.strip()
        elif self.role == "Dramaturg":
            directives["dramaturgical_notes"] = raw.strip()
        elif self.role == "Scene Writer":
            directives["next_scene_element"] = raw.strip()

        return directives


class Actor:
    def __init__(self, actor_name, gender):
        self.actor_name = actor_name
        self.gender = gender
        self.memory = []
        self.character = None

    def inhabit(self, character):
        self.character = character

    def think(self, cue_text, summary, crew_notes=None):
        if not self.character:
            return f"[ERROR] {self.actor_name} is not inhabiting any character."

        # Compile relevant crew notes
        compiled_notes = ""
        if crew_notes:
            if "props_to_add" in crew_notes:
                compiled_notes += f"Props: {crew_notes['props_to_add']}\n"
            if "stage_notes" in crew_notes:
                compiled_notes += f"Stage: {crew_notes['stage_notes']}\n"
            if "sound_effects" in crew_notes:
                compiled_notes += f"Sound: {crew_notes['sound_effects']}\n"
            if "lighting_changes" in crew_notes:
                compiled_notes += f"Lighting: {crew_notes['lighting_changes']}\n"

        prompt = f"""
You are {self.actor_name}, portraying {self.character['name']} ({self.character['role']}).

Character habits: {self.character['habits']}
Backstory: {self.character.get('backstory', 'Develop this as you go')}
Relationships: {self.character.get('relationships', 'Develop these as you go')}

Your recent acting choices: {self.memory[-3:] if self.memory else "First scene"}
Cue: {cue_text}
Crew Notes: {compiled_notes}
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

        # Store only the first line in memory to prevent it from getting too large
        first_line = result.splitlines()[0] if result.splitlines() else ""
        self.memory.append(first_line)
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

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


def generate_new_cue(scene_element, previous_cue, summary):
    """Generate a new cue based on the scene writer's input"""
    cue_prompt = f"""
    Previous cue: {previous_cue}
    Scene element to incorporate: {scene_element}
    Summary of last tick: {summary}

    Create a brief, evocative cue (1-2 sentences) that incorporates the scene element while 
    maintaining narrative continuity with the previous cue and summary.
    Focus on sensory details and atmosphere. 
    Do NOT include character dialogue or specific actions - only describe the environment and atmosphere.
    """

    new_cue = run_grok_prompt(cue_prompt, max_tokens=100)
    return new_cue.strip()


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
    {
        "name": "Lena", "gender": "female", "role": "Visionary",
        "habits": "Dreamy, intuitive",
        "backstory": "Has visions of the archive's past and possible futures. Struggles to distinguish between reality and her visions.",
        "relationships": "Spiritual connection with Orra, finds Jun's presence disruptive to her visions, often seeks Hale's protection when visions overwhelm her."
    },
    {
        "name": "Hale", "gender": "male", "role": "Guardian",
        "habits": "Protective, alert",
        "backstory": "Former military, hired to protect the archive but has developed a deeper connection to its mysteries than he lets on.",
        "relationships": "Protective of Lena, suspicious of Jun, respects Mara's knowledge, often clashes with Cellen's skepticism."
    },
    {
        "name": "Veera", "gender": "female", "role": "Historian",
        "habits": "Detail-oriented, persistent",
        "backstory": "Researching the archive's origins for her doctoral thesis, but has uncovered connections to her own ancestry.",
        "relationships": "Academic rivalry with Orra, fascinated by Lena's visions, respects Hale's pragmatism, finds Jun's antics distracting."
    },
    {
        "name": "Darik", "gender": "male", "role": "Engineer",
        "habits": "Practical, innovative",
        "backstory": "Called in to repair the archive's ancient systems, but has become entangled in its mysteries.",
        "relationships": "Works closely with Veera, annoyed by Jun's interference with the equipment, values Cellen's rationality, cautious around Mara."
    }
]

crew = [
    CrewMember("Salma Rayne", "Lighting Director"),
    CrewMember("Tarn Vale", "Cue Team Lead"),
    CrewMember("Greel", "Prop Master"),
    CrewMember("Echo Voss", "Sound Designer"),
    CrewMember("Merrin Hayes", "Stage Manager"),
    CrewMember("Dr. Laith Kouri", "Dramaturg"),
    CrewMember("Nova Penn", "Scene Writer")
]

actors = [
    Actor("Elias", "male"),
    Actor("Marcus", "male"),
    Actor("Anna", "female"),
    Actor("Elise", "female"),
    Actor("Sofia", "female"),
    Actor("Raymond", "male"),
    Actor("Aisha", "female"),
    Actor("Dominic", "male")
]

# Assign characters to actors
assigned = set()
for actor in actors:
    for character in characters:
        if character['gender'] == actor.gender and character['name'] not in assigned:
            actor.inhabit(character)
            assigned.add(character['name'])
            break

# Initial cue to get things started
initial_cue = "A low-frequency hum vibrates beneath the floorboards as the archive door shuts behind you."


def simulate_ticks(actors, crew, initial_cue, ticks=3):
    summary = ""
    tick_records = []
    current_cue = initial_cue

    for tick in range(ticks):
        print(f"\n====== TICK {tick + 1} ======")
        print(f"CURRENT CUE: {current_cue}")

        # First, get crew responses
        crew_responses = {}
        for member in crew:
            directives = member.act(current_cue, summary, tick + 1)
            print(f"[{member.name} - {member.role}]\n{json.dumps(directives, indent=2)}\n")
            crew_responses[member.name] = directives
            time.sleep(1)

        # Compile crew notes for actors
        compiled_crew_notes = {}
        for member_name, directives in crew_responses.items():
            compiled_crew_notes.update(directives)

        # Get actor responses
        actors_responses = {}
        character_names = []
        for actor in actors:
            if actor.character:
                character_names.append(actor.character['name'])

            # Use the cue from the Cue Team Lead if available, otherwise use the current cue
            cue_to_use = crew_responses.get("Tarn Vale", {}).get("cue_text_override", current_cue)

            result = actor.think(cue_to_use, summary, compiled_crew_notes)
            print(f"[{actor.actor_name} as {actor.character['name']} - {actor.character['role']}]\n{result.strip()}\n")
            actors_responses[actor.actor_name] = result
            time.sleep(1)

        # Enhance interactions
        if tick > 0:  # Skip first tick to establish characters
            interaction_suggestions = enhance_character_interactions(actors_responses, character_names)
            print(f"[INTERACTION SUGGESTIONS]\n{interaction_suggestions.strip()}\n")

        # Summarize the tick
        summary = summarize_tick(actors_responses, crew_responses)
        print(f"[TICK {tick + 1} SUMMARY]\n{summary.strip()}\n")

        # Generate the next cue using the Scene Writer's input
        if "Nova Penn" in crew_responses and "next_scene_element" in crew_responses["Nova Penn"]:
            scene_element = crew_responses["Nova Penn"]["next_scene_element"]
            current_cue = generate_new_cue(scene_element, current_cue, summary)
            print(f"[NEW CUE FOR NEXT TICK]\n{current_cue}\n")

        # Save tick data
        tick_data = {
            "tick": tick + 1,
            "cue": current_cue,
            "crew_responses": crew_responses,
            "actors_responses": actors_responses,
            "summary": summary,
            "next_cue": current_cue,
            "timestamp": datetime.now().isoformat()
        }
        tick_records.append(tick_data)

        # Periodically save simulation state
        if tick % 3 == 0 or tick == ticks - 1:
            with open(SAVE_FILE, "w") as f:
                json.dump(tick_records, f, indent=2)


simulate_ticks(actors, crew, initial_cue, ticks=30)