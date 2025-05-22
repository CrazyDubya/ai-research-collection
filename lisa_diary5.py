import json
import os
import random
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import textwrap
from openai import OpenAI

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
    """Generator for unexpected and interesting events with API enhancement"""

    def __init__(self):
        # Get API key from environment variable
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.evolution_counter = 0
        self.evolution_threshold = 5

        self.UNEXPECTED_EVENTS = [
            "accidentally trained an AI to write poetry about debugging errors",
            "discovered a correlation between coffee consumption and code quality",
            "found a pattern in city traffic that looks like a neural network",
            "started a underground tech meditation group"
        ]

        self.INTERESTING_HOBBIES = [
            "extreme debugging (debugging while rock climbing)",
            "quantum origami",
            "algorithmic gardening",
            "teaching meditation to robots"
        ]

        self.QUIRKY_THOUGHTS = [
            "What if consciousness is just a really well-optimized algorithm?",
            "Do computers dream of electric debugging?",
            "Maybe the universe is just one big neural network..."
        ]

    def generate_new_events(self, num_events: int = 3) -> List[str]:
        """Generate new events using the API"""
        event_examples = "\n".join(self.UNEXPECTED_EVENTS[:2])
        prompt = f"""Based on these example quirky tech events:
        {event_examples}
        Generate {num_events} new unique quirky tech events in a similar style."""

        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        new_events = response.choices[0].message.content.split("\n")
        new_events = [event.strip() for event in new_events if event.strip()]
        self.UNEXPECTED_EVENTS.extend(new_events)
        return new_events

    def generate_new_hobbies(self, num_hobbies: int = 2) -> List[str]:
        """Generate new hobbies using the API"""
        hobby_examples = "\n".join(self.INTERESTING_HOBBIES[:2])
        prompt = f"""Based on these example quirky tech hobbies:
        {hobby_examples}
        Generate {num_hobbies} new unique quirky tech hobbies in a similar style."""

        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        new_hobbies = response.choices[0].message.content.split("\n")
        new_hobbies = [hobby.strip() for hobby in new_hobbies if hobby.strip()]
        self.INTERESTING_HOBBIES.extend(new_hobbies)
        return new_hobbies

    def check_and_evolve(self):
        """Check if it's time to evolve content"""
        self.evolution_counter += 1
        if self.evolution_counter >= self.evolution_threshold:
            self.evolution_counter = 0
            return self.generate_new_events(), self.generate_new_hobbies()
        return None, None


class LisaLifeMemoryFaculty(CustomMentalFaculty):
    """Custom mental faculty for memory handling"""

    def __init__(self):
        super().__init__("Life Memory")
        self.add_actions({
            "REFLECT": {
                "description": "Reflect on past experiences and current scary situation",
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
        event = random.choice(agent.quirky_events.UNEXPECTED_EVENTS)
        agent.think(f"This reminds me of the time {event}")
        return True


class LisaWorld(TinyWorld):
    """Combined world for Lisa"""

    def __init__(self, name: str):
        super().__init__(name, agents=[])
        self.current_datetime = datetime.now()

        self.regular_locations = {
            "home_office": "A cozy space filled with monitors and plants",
            "cafe": "Local coffee shop with great wifi",
            "park": "Peaceful spot for midday walks"
        }

        self.quirky_locations = {
            "quantum_cafe": "A café where reality glitches occasionally",
            "debug_garden": "A garden where code grows like plants",
            "neural_nexus": "A space where thoughts become visible patterns"
        }

        self.locations = {**self.regular_locations, **self.quirky_locations}
def setup_diary_processor():
    """Set up the diary processor with proper export and enrichment"""
    exporter = ArtifactExporter(base_output_folder="./lisa_diary")
    enricher = TinyEnricher()
    return DiaryWriter(exporter=exporter, enricher=enricher)

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

def setup_combined_knowledge():
    """Create knowledge base combining practical and quirky content"""
    knowledge_base_dir = "./lisa_knowledge"
    os.makedirs(knowledge_base_dir, exist_ok=True)

    # Technical Knowledge
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
        self.diary_processor = setup_diary_processor()
        self.quirky_events = QuirkyEvent()
        self.load_lisa()
        self.world.add_agent(self.lisa)

    def load_lisa(self):
        """Initialize Lisa with combined traits"""
        self.lisa = TinyPerson(name="Lisa")
        self.lisa.quirky_events = self.quirky_events  # Give access to quirky events

        # Basic traits
        self.lisa.define("age", "32 (or 128 in binary)")
        self.lisa.define("nationality", "Digital Nomad")
        self.lisa.define("occupation", "Reality-Hacking Data Scientist")

        # Personality traits
        self.lisa.define_several("personality_traits", [
            "analytical and imaginative",
            "finds patterns everywhere",
            "talks to computers in poetry",
            "believes algorithms have feelings",
            "practices data-driven meditation"
        ])

        # Professional interests
        self.lisa.define_several("professional_interests", [
            "machine learning",
            "data visualization",
            "teaching meditation to neural networks",
            "finding consciousness in code"
        ])

        # Add mental faculties
        self.lisa.add_mental_faculties([
            RecallFaculty(),
            LisaLifeMemoryFaculty(),
            TinyToolUse(tools=[self.diary_processor])
        ])

        # Load knowledge base
        knowledge_dir = setup_combined_knowledge()
        self.lisa.read_documents_from_folder(knowledge_dir)

    def generate_life_slice(self, time_point: Optional[datetime] = None):
        """Generate a slice of Lisa's life"""
        if time_point is None:
            time_point = datetime.now() - timedelta(days=random.randint(0, 1825))

        # Update world's datetime
        self.world.current_datetime = time_point

        # Check for evolution of events
        new_events, new_hobbies = self.quirky_events.check_and_evolve()
        if new_events:
            print("\n=== Evolution in Lisa's Reality ===")
            print(f"New Events: {new_events}")
            print(f"New Hobbies: {new_hobbies}")

            # Add new location based on evolution
            if new_hobbies:
                new_location = f"The {random.choice(new_hobbies).split()[0]} Lab"
                self.world.quirky_locations[new_location] = f"A space dedicated to {new_hobbies[0]}"
                self.world.locations = {**self.world.regular_locations, **self.world.quirky_locations}

        # Decide if this slice should be quirky
        is_quirky = random.random() < 0.4  # 40% chance of quirky event

        if is_quirky:
            event = random.choice(self.quirky_events.UNEXPECTED_EVENTS)
            location = random.choice(list(self.world.quirky_locations.keys()))
        else:
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

        # Write diary entry
        self.diary_processor.write_diary_entry(
            timestamp=time_point,
            narrative=narrative,
            is_quirky=is_quirky,
            location=location
        )

        return narrative, is_quirky, location


def main():
    """Main execution function"""
    simulator = LisaLifeSimulator()

    print("\n=== Welcome to Lisa's Life Simulator ===")
    print(f"\nLisa's Current State:")
    print(f"Occupation: {simulator.lisa.get('occupation')}")
    print(f"Location: {random.choice(list(simulator.world.locations.items()))}")

    # Generate several life slices
    total_entries = 30
    while total_entries > 0:
        batch_size = random.randint(3, 7)
        current_batch = min(batch_size, total_entries)

        print(f"\n=== Generating {current_batch} Life Slices ===")
        for _ in range(current_batch):
            narrative, is_quirky, location = simulator.generate_life_slice()
            print(f"\n{'🌟 Quirky' if is_quirky else '💻 Normal'} Moment at {location}:")
            print(narrative)

            # Have Lisa reflect
            simulator.lisa.think_and_act(
                "IMAGINE connections between this experience and the nature of reality..."
                if is_quirky else
                "REFLECT on the practical implications..."
            )

        total_entries -= current_batch

        if total_entries > 0:
            print("\nProcessing reality shifts...")


if __name__ == "__main__":
    main()