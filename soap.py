import os
import shutil
import datetime
import random
from typing import Dict, List

from tinytroupe.agent import TinyPerson, RecallFaculty, CustomMentalFaculty
from tinytroupe.environment import TinyWorld
from tinytroupe.tools import TinyWordProcessor
from tinytroupe.story import TinyStory
from tinytroupe.extraction import ArtifactExporter

class QuirkyEvent:
    UNEXPECTED_EVENTS = [
        "accidentally trained an AI to write poetry about debugging errors",
        "discovered a correlation between coffee consumption and code quality",
        "found a pattern in city traffic that looks like a neural network",
        "started an underground tech meditation group"
    ]

    QUIRKY_THOUGHTS = [
        "What if consciousness is just a really well-optimized algorithm?",
        "Do computers dream of electric debugging?",
        "Maybe the universe is just one big neural network..."
    ]


class LisaLifeMemoryFaculty(CustomMentalFaculty):
    def __init__(self):
        super().__init__("Life Memory")
        self.add_actions({
            "REFLECT": {
                "description": "Reflect on past experiences",
                "function": self._process_reflection
            },
            "REMINISCE": {
                "description": "Recall and share a specific memory",
                "function": self._process_reminisce
            },
            "IMAGINE": {
                "description": "Create imaginative connections",
                "function": self._process_imagination
            }
        })

    def _process_reflection(self, agent, action):
        reflection = f"I find myself thinking about {action['content']}..."
        agent.think(reflection)
        if random.random() < 0.3:
            agent.think(random.choice(QuirkyEvent.QUIRKY_THOUGHTS))
        return True

    def _process_reminisce(self, agent, action):
        memory = f"I remember {action['content']}..."
        agent.think(memory)
        emotions = ["joy", "curiosity", "satisfaction", "wonder"]
        agent.think(f"That memory fills me with {random.choice(emotions)}...")
        return True

    def _process_imagination(self, agent, action):
        agent.think(f"What if... {action['content']}")
        event = random.choice(QuirkyEvent.UNEXPECTED_EVENTS)
        agent.think(f"This reminds me of the time I {event}")
        return True


class DocumentManager:
    """Each agent has exactly one folder with both 'public' and 'private' documents.
       Public documents are replicated across all agents.
       Private documents (like diaries) are only updated by the owner but stored in the same folder.
    """
    def __init__(self, base_dir="simulation_docs"):
        self.base_dir = base_dir
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        self.agent_dirs = {}
        self.public_documents = {}  # doc_name -> content
        self.diaries = {}  # agent_name -> [entries]

    def register_agent(self, agent_name: str):
        agent_dir = os.path.join(self.base_dir, agent_name)
        os.makedirs(agent_dir, exist_ok=True)
        self.agent_dirs[agent_name] = agent_dir
        self.diaries[agent_name] = []

    def _get_doc_path(self, agent_name: str, doc_name: str):
        return os.path.join(self.agent_dirs[agent_name], doc_name)

    def write_diary_entry(self, agent_name: str, entry: str):
        self.diaries[agent_name].append(entry)
        # Update diary file
        diary_path = self._get_doc_path(agent_name, "diary.md")
        with open(diary_path, "w") as f:
            for e in self.diaries[agent_name]:
                f.write(f"- {e}\n")

    def create_public_document(self, doc_name: str, content: str):
        self.public_documents[doc_name] = content
        self._replicate_public_document(doc_name)

    def update_public_document(self, doc_name: str, content: str):
        self.public_documents[doc_name] = content
        self._replicate_public_document(doc_name)

    def read_public_document(self, doc_name: str) -> str:
        return self.public_documents.get(doc_name, "")

    def _replicate_public_document(self, doc_name: str):
        """Write the updated public doc to all agents' directories."""
        content = self.public_documents[doc_name]
        for agent_dir in self.agent_dirs.values():
            doc_path = os.path.join(agent_dir, doc_name)
            with open(doc_path, "w") as f:
                f.write(content)

    def create_private_document(self, agent_name: str, doc_name: str, content: str):
        doc_path = self._get_doc_path(agent_name, doc_name)
        with open(doc_path, "w") as f:
            f.write(content)

    def update_private_document(self, agent_name: str, doc_name: str, content: str):
        self.create_private_document(agent_name, doc_name, content)

    def read_private_document(self, agent_name: str, doc_name: str) -> str:
        doc_path = self._get_doc_path(agent_name, doc_name)
        if not os.path.exists(doc_path):
            return ""
        with open(doc_path, "r") as f:
            return f.read()


def create_lisa():
    lisa = TinyPerson("Lisa")
    lisa.define("age", 28)
    lisa.define("nationality", "Canadian")
    lisa.define("occupation", "Data Scientist")
    lisa.define_several("personality_traits",
        [
            "Curious and loves to learn new things",
            "Analytical and likes to solve problems",
            "Friendly and enjoys sharing secrets",
            "Doesn't give up easily"
        ])
    lisa.add_mental_faculties([RecallFaculty(), LisaLifeMemoryFaculty()])
    return lisa

def create_oscar():
    oscar = TinyPerson("Oscar")
    oscar.define("age", 30)
    oscar.define("nationality", "German")
    oscar.define("occupation", "Architect")
    oscar.define_several("personality_traits",
        [
            "Fast paced, detail oriented",
            "Witty sense of humor",
            "Calm but can get extremely angry rarely"
        ])
    oscar.add_mental_faculties([RecallFaculty()])
    return oscar


class SoapOperaWorld(TinyWorld):
    def __init__(self, name: str, agents: List[TinyPerson], doc_manager: DocumentManager):
        super().__init__(name, agents)
        self.doc_manager = doc_manager
        self.concept = None

    def set_concept(self, concept: str):
        self.concept = concept

    def build_outline(self):
        outline = f"# Story Outline\nConcept: {self.concept}\n\n- Setting: TBD\n- Characters: TBD\n- Plot: TBD\n"
        # create public doc
        self.doc_manager.create_public_document("Outline.md", outline)

    def expand_outline_aspects(self, agent: TinyPerson):
        current_outline = self.doc_manager.read_public_document("Outline.md")
        agent.listen_and_act("RECALL 'outline inspiration'")
        expanded_outline = current_outline + "\n## Expanded Aspects\n- Setting: A bustling metropolis by the coast.\n- Characters: Conflicted but passionate professionals.\n- History: A long line of family feuds and rivalries.\n"
        self.doc_manager.update_public_document("Outline.md", expanded_outline)

    def build_chapter(self, agent: TinyPerson, chapter_num: int):
        outline = self.doc_manager.read_public_document("Outline.md")
        agent.listen(f"Use the outline to write Chapter {chapter_num} of the story based on concept: {self.concept}. Avoid clichés.")
        story = TinyStory(agent, purpose="Write a chapter", context=f"Chapter {chapter_num}")
        chapter_text = story.start_story(
            requirements="A well-structured narrative reflecting the outline and concept.",
            number_of_words=300,
            include_plot_twist=(chapter_num % 2 == 0)
        )
        chapter_name = f"Chapter_{chapter_num}.md"
        self.doc_manager.create_public_document(chapter_name, chapter_text)
        return chapter_text

    def agents_reflect_in_diaries(self, chapter_num: int):
        for agent in self.agents:
            entry = f"After writing chapter {chapter_num}, I feel..."
            if random.random() < 0.5:
                entry += f" Torn between my professional duties and my personal longing. The story's events mirror my own secret struggles..."
            else:
                entry += f" Proud and anxious. The chapter triggered memories of my past. I recall I once {random.choice(QuirkyEvent.UNEXPECTED_EVENTS)}."
            self.doc_manager.write_diary_entry(agent.name, entry)


def main():
    base_dir = "simulation_docs"
    doc_manager = DocumentManager(base_dir=base_dir)

    lisa = create_lisa()
    oscar = create_oscar()

    for a in [lisa, oscar]:
        doc_manager.register_agent(a.name)

    world = SoapOperaWorld("SoapOperaMix", agents=[lisa, oscar], doc_manager=doc_manager)

    concept = input("Enter the story concept: ")
    world.set_concept(concept)

    world.build_outline()
    world.expand_outline_aspects(lisa)

    num_chapters = 3
    for ch_num in range(1, num_chapters+1):
        chapter_text = world.build_chapter(oscar if ch_num % 2 == 0 else lisa, ch_num)
        print(f"\n--- Chapter {ch_num} ---\n{chapter_text}\n")
        world.agents_reflect_in_diaries(ch_num)

    print("Story and diaries generated. Each agent only sees one folder (their own), containing their diary and the public docs.")


if __name__ == "__main__":
    main()
