import json
import os
import random
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import textwrap

import chevron
import logging
import pandas as pd
import pypandoc
import markdown
from typing import Union

# TinyTroupe imports - ensure these are installed and accessible
from tinytroupe.agent import TinyPerson, RecallFaculty, CustomMentalFaculty
from tinytroupe.tools import TinyWordProcessor
from tinytroupe.extraction import ArtifactExporter
from tinytroupe.environment import TinyWorld
from tinytroupe.story import TinyStory
import tinytroupe.utils as utils
from tinytroupe.factory import TinyPersonFactory

from tinytroupe import openai_utils

logger = logging.getLogger("tinytroupe")


###########################################################################################
# ResultsExtractor class and related code (from the original snippet provided previously)
###########################################################################################
class ResultsExtractor:

    def __init__(self,
                 extraction_prompt_template_path: str = os.path.join(os.path.dirname(__file__),
                                                                     'prompts/interaction_results_extractor.mustache'),
                 extraction_objective: str = "The main points present in the agents' interactions history.",
                 situation: str = "",
                 fields: List[str] = None,
                 fields_hints: dict = None,
                 verbose: bool = False):
        self._extraction_prompt_template_path = extraction_prompt_template_path
        self.default_extraction_objective = extraction_objective
        self.default_situation = situation
        self.default_fields = fields
        self.default_fields_hints = fields_hints
        self.default_verbose = verbose
        self.agent_extraction = {}
        self.world_extraction = {}

    def extract_results_from_agents(self,
                                    agents: List[TinyPerson],
                                    extraction_objective: str = None,
                                    situation: str = None,
                                    fields: list = None,
                                    fields_hints: dict = None,
                                    verbose: bool = None):
        results = []
        for agent in agents:
            result = self.extract_results_from_agent(agent, extraction_objective, situation, fields, fields_hints,
                                                     verbose)
            results.append(result)

        return results

    def extract_results_from_agent(self,
                                   tinyperson: TinyPerson,
                                   extraction_objective: str = None,
                                   situation: str = None,
                                   fields: list = None,
                                   fields_hints: dict = None,
                                   verbose: bool = None):
        extraction_objective, situation, fields, fields_hints, verbose = self._get_default_values_if_necessary(
            extraction_objective, situation, fields, fields_hints, verbose
        )

        messages = []
        rendering_configs = {}
        if fields is not None:
            rendering_configs["fields"] = ", ".join(fields)

        if fields_hints is not None:
            rendering_configs["fields_hints"] = list(fields_hints.items())

        # Load the prompt template
        with open(self._extraction_prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        messages.append({"role": "system",
                         "content": chevron.render(prompt_template,
                                                   rendering_configs)})

        interaction_history = tinyperson.pretty_current_interactions(max_content_length=None)

        extraction_request_prompt = \
            f"""
## Extraction objective

{extraction_objective}

## Situation
You are considering a single agent, named {tinyperson.name}. Your objective thus refers to this agent specifically.
{situation}

## Agent Interactions History

You will consider an agent's history of interactions, which include stimuli it received as well as actions it 
performed.

{interaction_history}
"""
        messages.append({"role": "user", "content": extraction_request_prompt})

        next_message = openai_utils.client().send_message(messages, temperature=0.0)

        debug_msg = f"Extraction raw result message: {next_message}"
        logger.debug(debug_msg)
        if verbose:
            print(debug_msg)

        if next_message is not None:
            result = utils.extract_json(next_message["content"])
        else:
            result = None

        self.agent_extraction[tinyperson.name] = result
        return result

    def extract_results_from_world(self,
                                   tinyworld: TinyWorld,
                                   extraction_objective: str = "The main points that can be derived from the agents conversations and actions.",
                                   situation: str = "",
                                   fields: list = None,
                                   fields_hints: dict = None,
                                   verbose: bool = None):

        extraction_objective, situation, fields, fields_hints, verbose = self._get_default_values_if_necessary(
            extraction_objective, situation, fields, fields_hints, verbose
        )

        messages = []

        rendering_configs = {}
        if fields is not None:
            rendering_configs["fields"] = ", ".join(fields)

        if fields_hints is not None:
            rendering_configs["fields_hints"] = list(fields_hints.items())

        with open(self._extraction_prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        messages.append({"role": "system",
                         "content": chevron.render(prompt_template, rendering_configs)})

        interaction_history = tinyworld.pretty_current_interactions(max_content_length=None)

        extraction_request_prompt = \
            f"""
## Extraction objective

{extraction_objective}

## Situation
You are considering various agents.
{situation}

## Agents Interactions History

You will consider the history of interactions from various agents that exist in an environment called {tinyworld.name}. 
Each interaction history includes stimuli the corresponding agent received as well as actions it performed.

{interaction_history}
"""
        messages.append({"role": "user", "content": extraction_request_prompt})

        next_message = openai_utils.client().send_message(messages, temperature=0.0)

        debug_msg = f"Extraction raw result message: {next_message}"
        logger.debug(debug_msg)
        if verbose:
            print(debug_msg)

        if next_message is not None:
            result = utils.extract_json(next_message["content"])
        else:
            result = None

        self.world_extraction[tinyworld.name] = result
        return result

    def save_as_json(self, filename: str, verbose: bool = False):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"agent_extractions": self.agent_extraction,
                       "world_extraction": self.world_extraction}, f, indent=4)

        if verbose:
            print(f"Saved extraction results to {filename}")

    def _get_default_values_if_necessary(self,
                                         extraction_objective: str,
                                         situation: str,
                                         fields: List[str],
                                         fields_hints: dict,
                                         verbose: bool):

        if extraction_objective is None:
            extraction_objective = self.default_extraction_objective
        if situation is None:
            situation = self.default_situation
        if fields is None:
            fields = self.default_fields
        if fields_hints is None:
            fields_hints = self.default_fields_hints
        if verbose is None:
            verbose = self.default_verbose

        return extraction_objective, situation, fields, fields_hints, verbose


###########################################################################################
# Lisa's simulation code
###########################################################################################
@dataclass
class LifeEvent:
    timestamp: datetime
    event_type: str
    description: str
    mood: str
    location: str
    related_people: List[str]
    consequences: List[str]


class LisaWorld(TinyWorld):
    """A specialized world for Lisa's life simulation."""

    def __init__(self, name: str):
        super().__init__(name, agents=[])
        self.current_datetime = datetime.now()
        self.locations = {
            "home_office": "A cozy space filled with monitors and plants",
            "cafe": "Local coffee shop with great wifi",
            "park": "Peaceful spot for midday walks",
            "conference_room": "Virtual meeting space",
            "garden": "Small balcony garden with herbs"
        }
        self.current_events = []


class LisaLifeMemoryFaculty(CustomMentalFaculty):
    """A custom mental faculty for handling Lisa's life memories and experiences"""

    def __init__(self):
        super().__init__("Life Memory")
        self.add_actions({
            "REFLECT": {
                "description": "Reflect on past experiences and current situation",
                "function": self._process_reflection
            },
            "REMINISCE": {
                "description": "Actively recall and share a specific memory",
                "function": self._process_reminisce
            }
        })

        self.memory_prompts = [
            "coding breakthroughs",
            "team collaborations",
            "data insights",
            "personal growth moments",
            "creative inspirations"
        ]

    def _process_reflection(self, agent, action):
        reflection = f"I find myself thinking about {action['content']}..."
        agent.think(reflection)
        agent.think(f"This reminds me of how {random.choice(self.memory_prompts)} often leads to unexpected insights.")
        return True

    def _process_reminisce(self, agent, action):
        memory = f"I remember {action['content']}..."
        agent.think(memory)
        emotions = ["joy", "curiosity", "satisfaction", "wonder", "determination"]
        agent.think(f"That memory fills me with {random.choice(emotions)}...")
        return True


class LisaLifeSimulator:
    """Manages Lisa's life simulation using the TinyTroupe framework"""

    def __init__(self, timeline_file="lisa_timeline.json"):
        self.timeline_file = timeline_file
        self.world = LisaWorld("Lisa's World")
        self.load_lisa()
        self.world.add_agent(self.lisa)

    def load_lisa(self):
        """Initialize Lisa with appropriate traits and faculties"""
        self.lisa = TinyPerson(name="Lisa")

        # Basic traits
        self.lisa.define("age", "32")
        self.lisa.define("nationality", "American")
        self.lisa.define("occupation", "Data Scientist")
        self.lisa.define("country_of_residence", "United States")

        # Personality traits
        self.lisa.define_several("personality_traits", [
            "analytical",
            "creative",
            "empathetic",
            "curious about patterns in data and life",
            "sometimes gets lost in thought while coding",
            "finds beauty in clean code and elegant solutions",
            "enjoys explaining complex concepts simply"
        ])

        # Professional interests
        self.lisa.define_several("professional_interests", [
            "machine learning",
            "data visualization",
            "ethical AI",
            "statistical modeling",
            "automated decision systems",
            "natural language processing"
        ])

        # Personal interests
        self.lisa.define_several("personal_interests", [
            "hiking",
            "digital art",
            "reading science fiction",
            "growing herbs",
            "solving puzzles",
            "experimenting with generative art",
            "collecting vintage calculators"
        ])

        # Routines
        self.lisa.define_several("routines", [
            "Morning coffee while checking overnight model results",
            "Midday walk to clear her head",
            "Evening coding sessions with lo-fi music",
            "Weekend data visualization experiments",
            "Daily journal entries about interesting patterns discovered",
            "Virtual team standup meetings"
        ])

        # Skills
        self.lisa.define_several("skills", [
            "Python programming",
            "Statistical analysis",
            "Data visualization",
            "Machine learning",
            "Technical writing",
            "Project management",
            "Public speaking",
            "Mentoring junior data scientists"
        ])

        # Mental faculties
        self.lisa.add_mental_faculties([
            RecallFaculty(),
            LisaLifeMemoryFaculty()
        ])

    def generate_life_slice(self, time_point: Optional[datetime] = None) -> str:
        """Generate a slice of Lisa's life at a given time point"""
        if time_point is None:
            time_point = datetime.now() - timedelta(days=random.randint(0, 1825))

        self.world.current_datetime = time_point
        hour = time_point.hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 22:
            period = "evening"
        else:
            period = "late night"

        contexts = {
            "morning": [
                "Starting the day with fresh coffee and code review",
                "Early morning data analysis session",
                "Team standup preparation"
            ],
            "afternoon": [
                "Deep work session on machine learning models",
                "Collaborative problem-solving with team",
                "Client presentation preparation"
            ],
            "evening": [
                "Personal project exploration",
                "Learning new data science techniques",
                "Documentation and reflection time"
            ],
            "late night": [
                "Late night debugging session",
                "Inspiration-driven coding",
                "Quiet contemplation of complex problems"
            ]
        }

        context = random.choice(contexts[period])
        self.lisa.define("current_datetime", time_point.isoformat())
        self.lisa.define("current_location", random.choice(list(self.world.locations.keys())))
        self.lisa.define("current_context", context)

        story = TinyStory(
            agent=self.lisa,
            purpose=f"Experience a flirty moment in Lisa's life during {period}",
            context=f"It's {period} and Lisa is {context.lower()}."
        )

        self.lisa.think_and_act(f"It's {period} and I'm {context.lower()}...")

        narrative = story.start_story(
            requirements=f"Write about Lisa's experiences during this {period}, focusing on her thoughts, feelings, and interactions with data and code.",
            number_of_words=300,
            include_plot_twist=random.random() < 0.3
        )

        self.lisa.think_and_act("This moment makes me reflect on my journey in data science...")

        continuation = story.continue_story(
            requirements="Add Lisa's internal monologue about data patterns and human behavior",
            number_of_words=400
        )

        return f"{narrative}\n\n{continuation}"

    def run_simulation(self, duration_days: int = 1):
        """Run the simulation for a specified number of days"""
        start_time = self.world.current_datetime
        end_time = start_time + timedelta(days=duration_days)

        while self.world.current_datetime < end_time:
            self.world._step(timedelta_per_step=timedelta(hours=3))
            self.lisa.think_and_act("Let me reflect on recent events...")
            actions = self.lisa.pop_latest_actions()
            for action in actions:
                if action['type'] in ['REFLECT', 'REMINISCE']:
                    print(f"\nLisa's Reflection: {action['content']}")


def main():
    """Main execution function for Lisa's life simulation"""
    simulator = LisaLifeSimulator()

    print("\n=== Initializing Lisa's Life Simulator ===")
    print(f"\nLisa's Configuration:")
    print(f"Occupation: {simulator.lisa.get('occupation')}")
    print(f"Age: {simulator.lisa.get('age')}")
    print(f"Location: {simulator.lisa.get('current_location')}")

    # Generate a slice of Lisa's life
    print("\nGenerating a slice of Lisa's life...")
    narrative = simulator.generate_life_slice()

    print("\n=== A Moment in Lisa's Life ===")
    print(narrative)

    # Run a short simulation
    print("\n=== Running a brief simulation ===")
    simulator.run_simulation(duration_days=1)

    # After simulation, we perform extraction
    extractor = ResultsExtractor(
        # Adjust the path to your prompt template if needed
        extraction_prompt_template_path=os.path.join(os.path.dirname(__file__),
                                                     'prompts/interaction_results_extractor.mustache')
    )

    # 1. Extract entire chat from Lisa
    lisa_full_extraction = extractor.extract_results_from_agent(
        simulator.lisa,
        extraction_objective="Extract the main points from Lisa's entire recent interaction history."
    )
    extractor.save_as_json("lisa_full_extraction.json")
    print("Full extraction saved to lisa_full_extraction.json")

    # 2. Extract last N statements by Lisa
    last_n = 5
    all_interactions = simulator.lisa.episodic_memory.retrieve_all()
    last_n_interactions = all_interactions[-last_n:]

    print(f"\n=== Last {last_n} Interactions by Lisa ===")
    for i, msg in enumerate(last_n_interactions, 1):
        print(f"{i}. {msg}")

    # Temporarily override pretty_current_interactions to return only last N interactions
    original_method = simulator.lisa.pretty_current_interactions

    def last_n_pretty_interactions(max_content_length=None):
        formatted = []
        for m in last_n_interactions:
            role = m.get("role", "unknown")
            content = m.get("content", {})
            formatted.append(f"Role: {role}\nContent: {content}\n")
        return "\n".join(formatted)

    simulator.lisa.pretty_current_interactions = last_n_pretty_interactions

    # Extract results for just the last N messages
    lisa_last_n_extraction = extractor.extract_results_from_agent(
        simulator.lisa,
        extraction_objective=f"Extract the main points from Lisa's last {last_n} interactions."
    )

    # Restore original method
    simulator.lisa.pretty_current_interactions = original_method

    extractor.save_as_json("lisa_last_n_extraction.json")
    print(f"Last {last_n} interactions extraction saved to lisa_last_n_extraction.json")


if __name__ == "__main__":
    main()
