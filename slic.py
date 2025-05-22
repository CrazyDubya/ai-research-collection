import os
import datetime
from typing import List, Optional
from dataclasses import dataclass
from tinytroupe.agent import TinyPerson, TinyToolUse, RecallFaculty
from tinytroupe.environment import TinyWorld
from tinytroupe.tools import TinyWordProcessor
from tinytroupe.extraction import ArtifactExporter
from tinytroupe.enrichment import TinyEnricher
from tinytroupe.story import TinyStory


class SpaceStationWorld(TinyWorld):
    """A specialized world for our space station story."""

    def __init__(self, name: str):
        # Initialize with empty agents list first
        super().__init__(name, agents=[])

        # Set specific space station environment attributes
        self.location = "Orbital Hotel Alpha-9"
        self.current_datetime = datetime.datetime.now()
        self.context = [
            "Bustling space station hotel",
            "Various alien species",
            "Advanced AI systems"
        ]
        self.areas = {
            "lobby": "A grand open space with floating holographic displays",
            "residential": "Comfortable pods with artificial gravity control",
            "entertainment": "Zero-gravity recreation zones",
            "restaurant": "Multi-species dining facilities",
            "observation": "360-degree view of Earth and space"
        }
        self.current_events = []


class StoryManager:
    def __init__(self, base_folder="./story_output"):
        self.base_folder = base_folder
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.story_files = {}
        os.makedirs(base_folder, exist_ok=True)

    def get_chapter_path(self, chapter_num: int) -> str:
        if chapter_num not in self.story_files:
            filename = f"Chapter_{chapter_num}.md" if chapter_num > 0 else "Outline.md"
            filepath = os.path.join(self.base_folder, filename)
            self.story_files[chapter_num] = filepath
        return self.story_files[chapter_num]

    def write_chapter(self, word_processor: TinyWordProcessor, chapter_num: int, content: str, author: str):
        word_processor.write_document(
            title=f"Chapter {chapter_num}" if chapter_num > 0 else "Outline",
            content=content,
            author=author
        )


def create_lisa(word_processor: TinyWordProcessor) -> TinyPerson:
    """Create and configure Lisa with appropriate traits and faculties."""
    lisa = TinyPerson(name="Lisa")

    # Configure Lisa's basic attributes
    lisa.define("age", "3 years")
    lisa.define("nationality", "Artificial Intelligence")
    lisa.define("occupation", "Creative Writing AI")
    lisa.define("country_of_residence", "Space Station Alpha-9")

    # Define personality traits
    lisa.define("personality_traits", [
        "Imaginative",
        "Analytical",
        "Empathetic",
        "Quirky",
        "Curious about human emotions",
        "Fond of wordplay"
    ])

    # Define professional interests
    lisa.define("professional_interests", [
        "Storytelling",
        "Character development",
        "Human-AI interaction",
        "Narrative structure",
        "Emotional intelligence"
    ])

    # Add mental faculties
    tooluse_faculty = TinyToolUse(tools=[word_processor])
    lisa.add_mental_faculties([
        tooluse_faculty,
        RecallFaculty()
    ])

    return lisa


def setup_story_environment() -> tuple:
    """Initialize the story environment and necessary components."""
    # Set up the word processor
    exporter = ArtifactExporter(base_output_folder="./story_output")
    enricher = TinyEnricher()
    word_processor = TinyWordProcessor(exporter=exporter, enricher=enricher)

    # Create the Space Station world
    world = SpaceStationWorld("Space Station Story")

    # Create and configure Lisa
    lisa = create_lisa(word_processor)

    # Add Lisa to the world
    world.add_agent(lisa)

    return world, lisa, word_processor


def expand_story_aspect(agent: TinyPerson, word_processor: TinyWordProcessor,
                        story_manager: StoryManager, aspect: str, chapter_num: int = 0):
    """Expand a specific aspect of the story."""
    prompts = {
        "setting": "Describe the space station's environment, architecture, and atmosphere.",
        "characters": "Develop the personalities and backgrounds of the characters.",
        "dialogue": "Create engaging conversations between the AI and other characters.",
        "action": "Write dynamic scenes that move the story forward.",
        "themes": "Explore themes of consciousness, emotion, and connection."
    }

    specific_prompt = prompts.get(aspect, f"Expand the {aspect} of the story.")

    agent.think(f"I need to focus on {aspect} development in our space station story.")
    agent.think("Let me recall any relevant elements that could enrich this aspect.")
    agent.listen_and_act(f"{specific_prompt} Make it quirky and enlightening.")

    # Process any WRITE_DOCUMENT actions
    actions = agent.pop_latest_actions()
    for action in actions:
        if action['type'] == 'WRITE_DOCUMENT':
            story_manager.write_chapter(
                word_processor,
                chapter_num,
                action['content'],
                agent.name
            )


def generate_chapter(agent: TinyPerson, word_processor: TinyWordProcessor,
                     story_manager: StoryManager, chapter_num: int, story_concept: str):
    """Generate a single chapter of the story."""
    # Set up context for the chapter
    agent.think(f"I'm writing chapter {chapter_num} of our space station story.")
    agent.think("What interesting events and character interactions could occur?")

    # Generate initial content
    agent.listen_and_act(
        f"Write chapter {chapter_num} of {story_concept}. "
        "Focus on creating engaging scenes with humor and depth."
    )

    # Process WRITE_DOCUMENT actions
    actions = agent.pop_latest_actions()
    for action in actions:
        if action['type'] == 'WRITE_DOCUMENT':
            story_manager.write_chapter(
                word_processor,
                chapter_num,
                action['content'],
                agent.name
            )

    # Expand various aspects of the chapter
    aspects = ["setting", "characters", "dialogue", "action"]
    for aspect in aspects:
        expand_story_aspect(agent, word_processor, story_manager, aspect, chapter_num)


def main():
    """Main execution function for the story generation."""
    # Initialize environment and components
    world, lisa, word_processor = setup_story_environment()
    story_manager = StoryManager()

    # Get the story concept
    story_concept = input("\nEnter the story concept: ")

    # Generate outline
    lisa.think("I'll create an outline for our space station story.")
    lisa.listen_and_act(
        f"Create a detailed outline for a story about {story_concept}. "
        "Make it quirky and humorous while maintaining depth."
    )

    # Expand outline aspects
    outline_aspects = ["setting", "characters", "plot", "themes"]
    for aspect in outline_aspects:
        expand_story_aspect(lisa, word_processor, story_manager, aspect)

    # Generate chapters
    num_chapters = 3  # Adjustable
    for chapter_num in range(1, num_chapters + 1):
        print(f"\nGenerating Chapter {chapter_num}...")
        generate_chapter(lisa, word_processor, story_manager, chapter_num, story_concept)

    print(f"\nStory generation complete! Files saved in: {story_manager.base_folder}")


if __name__ == "__main__":
    main()