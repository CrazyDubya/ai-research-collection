import json
import os
import random
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import textwrap

from tinytroupe.agent import TinyPerson, RecallFaculty, CustomMentalFaculty, TinyToolUse
from tinytroupe.tools import TinyWordProcessor
from tinytroupe.extraction import ArtifactExporter
from tinytroupe.environment import TinyWorld
from tinytroupe.story import TinyStory
from tinytroupe.enrichment import TinyEnricher
import tinytroupe.utils as utils
@dataclass
class LifeEvent:
    timestamp: datetime
    event_type: str
    description: str
    mood: str
    location: str
    related_people: List[str]
    consequences: List[str]
class QuirkyEvent:
    """Generator for unexpected and interesting events"""
    UNEXPECTED_EVENTS = [
        "accidentally trained an AI to write poetry about debugging errors",
        "discovered a correlation between coffee consumption and code quality",
        "found a pattern in city traffic that looks like a neural network",
        "started a underground tech meditation group"
    ]

    INTERESTING_HOBBIES = [
        "extreme debugging (debugging while rock climbing)",
        "quantum origami",
        "algorithmic gardening",
        "teaching meditation to robots"
    ]

    QUIRKY_THOUGHTS = [
        "What if consciousness is just a really well-optimized algorithm?",
        "Do computers dream of electric debugging?",
        "Maybe the universe is just one big neural network..."
    ]


class LisaLifeMemoryFaculty(CustomMentalFaculty):
    """Custom mental faculty combining normal and quirky memory handling"""

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
            },
            "IMAGINE": {
                "description": "Create imaginative connections between concepts",
                "function": self._process_imagination
            }
        })
        self.memory_prompts = [
            "coding breakthroughs",
            "team collaborations",
            "data insights",
            "unexpected discoveries"
        ]

    def _process_reflection(self, agent, action):
        reflection = f"I find myself thinking about {action['content']}..."
        agent.think(reflection)
        if random.random() < 0.3:  # 30% chance of quirky thought
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
        agent.think(f"This reminds me of the time {event}")
        return True


class LisaWorld(TinyWorld):
    """Combined normal and quirky world for Lisa"""

    def __init__(self, name: str):
        super().__init__(name, agents=[])
        self.current_datetime = datetime.now()

        # Normal locations
        self.regular_locations = {
            "home_office": "A cozy space filled with monitors and plants",
            "cafe": "Local coffee shop with great wifi",
            "park": "Peaceful spot for midday walks"
        }

        # Quirky locations
        self.quirky_locations = {
            "quantum_cafe": "A café where reality glitches occasionally",
            "debug_garden": "A garden where code grows like plants",
            "neural_nexus": "A space where thoughts become visible patterns"
        }

        # Combined locations
        self.locations = {**self.regular_locations, **self.quirky_locations}
        self.current_events = []


def setup_combined_knowledge():
    """Create knowledge base combining practical and quirky content"""
    knowledge_base_dir = "./lisa_knowledge"
    os.makedirs(knowledge_base_dir, exist_ok=True)

    # Technical Knowledge (from original)
    with open(f"{knowledge_base_dir}/technical_knowledge.md", "w") as f:
        f.write("""# Technical Knowledge and Skills

## Programming
- Expert in Python for data science
- Proficient in SQL and database optimization
- Experience with distributed computing
- Version control with Git

## Data Science
- Machine Learning algorithms and implementations
- Statistical analysis and hypothesis testing
- Data visualization techniques
- Neural network architectures
""")

    # Quirky Knowledge
    with open(f"{knowledge_base_dir}/unexpected_adventures.md", "w") as f:
        f.write("""# Unexpected Adventures and Discoveries

## Strange Discoveries
- Found patterns in city noise that matched stock market trends
- Discovered my plants grow better when I debug code nearby
- Realized my cat can predict system crashes

## Weird Experiments
- Tried teaching my AI to understand cat language
- Attempted to generate music from bug reports
- Created a meditation app that talks in SQL queries
""")

    # Combined Personal History
    with open(f"{knowledge_base_dir}/personal_history.md", "w") as f:
        f.write("""# Personal Background and Experiences

## Professional Life
- PhD in Computer Science with focus on Machine Learning
- Research in pattern recognition algorithms
- Leading innovative data science projects

## Alternative Pursuits
- Founded the 'Binary Beatniks' - poets who code in rhyme
- Started 'Silent Debugging' - meditation meets programming
- Created 'Algorithm Theatre' - where code comes alive

## Personal Growth
- Learning to explain complex concepts simply
- Finding balance between analysis and intuition
- Discovering patterns in unexpected places
""")

    return knowledge_base_dir


class LisaLifeSimulator:
    def __init__(self, timeline_file="lisa_timeline.json"):
        self.timeline_file = timeline_file
        self.world = LisaWorld("Lisa's Quantum Reality")
        self.load_lisa()
        self.world.add_agent(self.lisa)

    def load_lisa(self):
        """Initialize Lisa with both practical and quirky traits"""
        self.lisa = TinyPerson(name="Lisa")

        # Basic traits with a twist
        self.lisa.define("age", "32 (or 128 in binary)")
        self.lisa.define("nationality", "Digital Nomad")
        self.lisa.define("occupation", "Reality-Hacking Data Scientist")

        # Combined personality traits
        self.lisa.define_several("personality_traits", [
            "analytical and imaginative",
            "finds patterns everywhere",
            "talks to computers in poetry",
            "believes algorithms have feelings",
            "practices data-driven meditation"
        ])

        # Professional and quirky interests
        self.lisa.define_several("professional_interests", [
            "machine learning",
            "data visualization",
            "teaching meditation to neural networks",
            "finding consciousness in code"
        ])

        # Add mental faculties
        self.lisa.add_mental_faculties([
            RecallFaculty(),
            LisaLifeMemoryFaculty()
        ])

        # Load knowledge base
        knowledge_dir = setup_combined_knowledge()
        self.lisa.read_documents_from_folder(knowledge_dir)

    def generate_life_slice(self, time_point: Optional[datetime] = None):
        """Generate a slice of Lisa's life combining normal and quirky events"""
        if time_point is None:
            time_point = datetime.now() - timedelta(days=random.randint(0, 1825))

        # Update world's datetime
        self.world.current_datetime = time_point

        # Decide if this slice should be quirky or normal
        is_quirky = random.random() < 0.4  # 40% chance of quirky event

        if is_quirky:
            event = random.choice(QuirkyEvent.UNEXPECTED_EVENTS)
            location = random.choice(list(self.world.quirky_locations.keys()))
        else:
            # Generate a normal work/life event
            events = [
                "debugging a complex algorithm",
                "mentoring a junior data scientist",
                "optimizing database queries",
                "writing documentation"
            ]
            event = random.choice(events)
            location = random.choice(list(self.world.regular_locations.keys()))

        # Update Lisa's state
        self.lisa.define("current_datetime", time_point.isoformat())
        self.lisa.define("current_location", location)
        self.lisa.define("current_context", event)

        # Generate narrative
        story = TinyStory(
            agent=self.lisa,
            purpose="Share an experience",
            context=f"Experiencing: {event}"
        )

        # Have Lisa reflect on the experience
        self.lisa.think(f"Something interesting is happening: {event}")
        self.lisa.listen_and_act("RECALL similar experiences")

        narrative = story.start_story(
            requirements="Describe this experience, whether mundane or extraordinary",
            number_of_words=300,
            include_plot_twist=is_quirky
        )

        return narrative, is_quirky


class DiaryWriter(TinyWordProcessor):
    """Enhanced word processor for Lisa's diary entries"""

    def __init__(self, owner=None, exporter=None, enricher=None):
        super().__init__(owner=owner, exporter=exporter, enricher=enricher)
        self.diary_index = []

    def write_diary_entry(self, timestamp: datetime, narrative: str, is_quirky: bool, location: str):
        """Write a formatted diary entry"""
        title = f"Diary Entry - {timestamp.strftime('%Y-%m-%d %H:%M')}"

        # Format the diary entry with markdown
        content = f"""# {title}

## Location
{location}

## Mood
{'🌟 Feeling Whimsical' if is_quirky else '💻 Focused and Analytical'}

## Experience
{narrative}

---
*Written by Lisa, your friendly neighborhood reality-hacking data scientist*
"""
        # Save using the word processor's functionality
        self.write_document(
            title=title,
            content=content,
            author="Lisa"
        )

        # Keep track of entries
        self.diary_index.append({
            'timestamp': timestamp,
            'location': location,
            'is_quirky': is_quirky,
            'title': title
        })


def setup_diary_processor():
    """Set up the diary processor with proper export and enrichment"""
    exporter = ArtifactExporter(base_output_folder="./lisa_diary")
    enricher = TinyEnricher()
    return DiaryWriter(exporter=exporter, enricher=enricher)


class LisaLifeSimulator:
    def __init__(self, timeline_file="lisa_timeline.json"):
        self.timeline_file = timeline_file
        self.world = LisaWorld("Lisa's Quantum Reality")
        self.diary_processor = setup_diary_processor()
        self.load_lisa()
        self.world.add_agent(self.lisa)

    def load_lisa(self):
        """Initialize Lisa with both practical and quirky traits"""
        self.lisa = TinyPerson(name="Lisa")

        # Basic traits with a twist
        self.lisa.define("age", "32 (or 128 in binary)")
        self.lisa.define("nationality", "Digital Nomad")
        self.lisa.define("occupation", "Reality-Hacking Data Scientist")

        # Combined personality traits
        self.lisa.define_several("personality_traits", [
            "analytical and imaginative",
            "finds patterns everywhere",
            "talks to computers in poetry",
            "believes algorithms have feelings",
            "practices data-driven meditation",
            "loves to debug reality",
            "dreams in code"
        ])

        # Professional and quirky interests
        self.lisa.define_several("professional_interests", [
            "machine learning",
            "data visualization",
            "teaching meditation to neural networks",
            "finding consciousness in code"
        ])

        # Add mental faculties with diary processor
        self.lisa.add_mental_faculties([
            RecallFaculty(),
            LisaLifeMemoryFaculty(),
            TinyToolUse(tools=[self.diary_processor])
        ])

        # Load knowledge base
        knowledge_dir = setup_combined_knowledge()
        self.lisa.read_documents_from_folder(knowledge_dir)

    def generate_life_slice(self, time_point: Optional[datetime] = None):
        """Generate a slice of Lisa's life combining normal and quirky events"""
        if time_point is None:
            time_point = datetime.now() - timedelta(days=random.randint(0, 1825))

        # Update world's datetime
        self.world.current_datetime = time_point

        # Decide if this slice should be quirky or normal
        is_quirky = random.random() < 0.4  # 40% chance of quirky event

        if is_quirky:
            event = random.choice(QuirkyEvent.UNEXPECTED_EVENTS)
            location = random.choice(list(self.world.quirky_locations.keys()))
        else:
            events = [
                "debugging a complex algorithm",
                "mentoring a junior data scientist",
                "optimizing database queries",
                "writing documentation",
                "inventing a new product",
                "coding a new algorithm",
                "flirting at the coffee shop",
                "caught in the closet with a colleague"
            ]
            event = random.choice(events)
            location = random.choice(list(self.world.regular_locations.keys()))

        # Update Lisa's state
        self.lisa.define("current_datetime", time_point.isoformat())
        self.lisa.define("current_location", location)
        self.lisa.define("current_context", event)

        # Generate narrative
        story = TinyStory(
            agent=self.lisa,
            purpose="Share an experience in your personal life",
            context=f"Experiencing: {event}"
        )

        # Have Lisa reflect on the experience
        self.lisa.think(f"Something interesting is happening: {event}")
        self.lisa.listen_and_act("RECALL similar experiences")

        narrative = story.start_story(
            requirements="Describe this experience, whether mundane or extraordinary",
            number_of_words=400,
            include_plot_twist=is_quirky
        )

        # Write diary entry
        self.diary_processor.write_diary_entry(
            timestamp=time_point,
            narrative=narrative,
            is_quirky=is_quirky,
            location=location
        )

        return narrative, is_quirky, location


def main():
    """Main execution function for Lisa's life simulation"""
    simulator = LisaLifeSimulator()

    print("\n=== Welcome to Lisa's Life Simulator ===")
    print(f"\nLisa's Current State:")
    print(f"Occupation: {simulator.lisa.get('occupation')}")
    print(f"Location: {random.choice(list(simulator.world.locations.items()))}")

    # Generate several life slices
    for i in range(30):
        print(f"\n=== Life Slice {i + 1} ===")
        narrative, is_quirky, location = simulator.generate_life_slice()
        print(f"{'🌟 Quirky Moment' if is_quirky else '💻 Normal Moment'} at {location}:")
        print(narrative)

        # Have Lisa reflect and document
        simulator.lisa.think_and_act(
            "IMAGINE connections between this experience and the nature of reality..."
            if is_quirky else
            "REFLECT on the practical implications..."
        )

        print(f"\nDiary entry saved for {simulator.world.current_datetime.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()