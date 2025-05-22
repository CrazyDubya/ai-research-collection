from examples import *

import json
import sys
import csv
import random
import tinytroupe
from tinytroupe.openai_utils import force_api_type
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.agent import *
from tinytroupe.environment import TinyWorld
from tinytroupe import control
from tinytroupe.extraction import ResultsExtractor, ResultsReducer
from tinytroupe.enrichment import TinyEnricher
from tinytroupe.extraction import ArtifactExporter
from tinytroupe.tools import TinyWordProcessor
from tinytroupe.story import TinyStory
import tinytroupe.utils as utils
from examples import *
# Set up export folder
data_export_folder = "data/byhand"

# Initialize components
exporter = ArtifactExporter(base_output_folder=data_export_folder)
enricher = TinyEnricher()
word_processor = TinyWordProcessor(exporter=exporter, enricher=enricher)  # Define the word processor
tooluse_faculty = TinyToolUse(tools=[word_processor])  # Pass the word processor to the faculty

# Create agents
oscar = create_oscar_the_architect()
lisa = create_lisa_the_data_scientist()
marcos = create_marcos_the_physician()
lila = create_lila_the_linguist()
sadie = create_sadie_the_model()
roxy = create_roxy_the_secretary()
mona = create_mistress_mona()
candy = create_candy_the_coed()
honey = create_honey_the_student()
aiden = create_aiden_the_AI_researcher()
seraphina = create_seraphina_the_environmental_diplomat()
kazuo = create_kazuo_the_vr_game_developer()


class LisaLifeMemoryFaculty(CustomMentalFaculty):
    def __init__(self):
        super().__init__("Life Memory")
        self.add_actions({
            "IMAGINE": {
                "description": "Create imaginative connections",
                "function": self._process_imagination
            }
        })

    def _process_imagination(self, agent, action):
        agent.think("I'm imagining " + action['content'])
        return True


# Define a chat room scenario
all_agents = [
    oscar, lisa, marcos, lila,
    sadie, roxy, mona,
    candy, honey, aiden, seraphina, kazuo
]

chat_room = TinyWorld("UN Security Council", all_agents)
chat_room.make_everyone_accessible()

# Randomly select a subset of agents that will form definitive opinions.
# The remaining agents will debate them.
num_agents = len(all_agents)
# Choose a random number k in [2, num_agents-1]
k = random.randint(2, num_agents - 1)
# Randomly select k agents to be the "concrete opinion givers"
opinion_formers = random.sample(all_agents, k)
# The rest will be debaters
debaters = [agent for agent in all_agents if agent not in opinion_formers]

# Construct instructions emphasizing the new objective:
# - The chosen subset (opinion_formers) should produce concrete, detailed proposals and suggestions.
# - The remainder (debaters) should challenge, debate, or question those proposals.
# - Emphasize clarity, severity, and encourage them to not be vague or boring.
instructions = f"""
Below are updated instructions for this run of the world (the UN Security Council scenario):

We have {num_agents} participants. Among them, {k} will form definitive, concrete opinions and make strong, well-defined suggestions about resource deployment and strategies. The remaining {num_agents - k} participants will debate, question, and challenge these suggestions, ensuring rigorous scrutiny.

**Key Objective:**
- The selected {k} agents must provide clear, concrete, and detailed proposals. They should not be vague. They must account for the severity of the tasks: large sums of money, troops, and the stakes of international security. They should break down spending plans, troop allocations, and strategic timelines.
- The other {num_agents - k} agents must actively debate these suggestions, either by proposing alternatives, pointing out flaws, or demanding more justification. They must not remain passive or overly agreeable. They should push for details, contest assumptions, and highlight potential risks.
- Everyone should avoid being too vague or repetitive. The gravity of the situation demands substance, specificity, and real scrutiny of proposals.

**Roles:**
- These agents form definitive opinions and propose concrete solutions:
  {[agent.name for agent in opinion_formers]}

- These agents debate, question, and challenge the proposals:
  {[agent.name for agent in debaters]}

**Available Actions (reminder):**
- RECALL: Retrieve info from memory
- CONSULT: Access a specific document (after LIST_DOCUMENTS)
- LIST_DOCUMENTS: See which documents are available
- THINK: Reflect internally
- TALK: Speak to agents or environment
- DONE: Conclude action sequence

Use these actions according to their constraints and provide meaningful, scenario-appropriate content. Do not remain boring or vague. Consider the severity of global security issues and make tangible suggestions and critiques.
"""

# Broadcast the new instructions
chat_room.broadcast(instructions)
chat_room.broadcast(
    "The UN Security Council convenes to discuss global security issues. Given the resources at hand (1 Billion Dollars, 10,000 non-combat troops, 1,000 combat troops), the selected opinion-makers must propose concrete, strategic allocations and plans. The others must rigorously debate and scrutinize these suggestions.")

# Run the scenario for a bit
chat_room.run_hours(1)

# Let's pick one of the opinion formers to prompt them for a proposal
# (If needed, choose a random opinion former)
main_proposer = random.choice(opinion_formers)
main_proposer.listen_and_act(instructions,
                             "Please draft a specific proposal covering the use of all resources, including detailed spending allocations and troop deployments, and outline your strategic reasoning.")
chat_room.run_hours(1)

# After the main proposer speaks, others will debate
for debater in debaters:
    debater.listen_and_act(instructions,
                           f"{main_proposer.name} has made a proposal. Debate it. Demand specifics, question assumptions, or offer alternative strategies.")
chat_room.run_hours(1)

# Optionally, run another hour to see continued interactions
chat_room.run_hours(1)

# The simulation should now reflect the updated objectives.
