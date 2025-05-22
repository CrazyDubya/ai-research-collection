import json
import os
import random
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import textwrap

from tinytroupe.agent import TinyPerson, RecallFaculty, CustomMentalFaculty
from tinytroupe.tools import TinyWordProcessor
from tinytroupe.extraction import ArtifactExporter
from tinytroupe.environment import TinyWorld
from tinytroupe.story import TinyStory
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


class LisaWorld(TinyWorld):
    """A specialized world for Lisa's life simulation."""   """A specialized world for Lisa's life simulation."""

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


def setup_lisa_knowledge():
    """Create knowledge base documents for Lisa's semantic memory."""
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
- Feature engineering methods

## Project Experience
- Built recommendation systems
- Implemented natural language processing pipelines
- Developed automated reporting systems
- Created real-time analytics dashboards
""")

    # Personal History
    with open(f"{knowledge_base_dir}/personal_history.md", "w") as f:
        f.write("""# Personal Background and Experiences

## Education
- PhD in Computer Science with focus on Machine Learning
- Research in pattern recognition algorithms
- Teaching assistant for introductory programming

## Key Life Events
- First programming experience at age 12
- Won national coding competition in college
- Published research on efficient algorithm design
- Mentored junior data scientists
- Started meditation practice to balance technical work

## Personal Growth
- Learning to explain complex concepts simply
- Finding balance between analysis and intuition
- Developing leadership skills
- Combining creativity with technical precision
""")

    # Current Projects
    with open(f"{knowledge_base_dir}/current_projects.md", "w") as f:
        f.write("""# Current Projects and Interests

## Work Projects
- Developing new recommendation algorithm
- Optimizing data processing pipeline
- Mentoring junior team members
- Research into ethical AI practices

## Personal Projects
- Creating generative art with Python
- Building a home automation system
- Writing technical blog posts
- Contributing to open source projects

## Learning Goals
- Advanced GPU computing
- Quantum computing basics
- Leadership and team management
- Technical writing improvement
""")

    return knowledge_base_dir


class LisaLifeSimulator:
    """Manages Lisa's life simulation using the TinyTroupe framework"""

    def __init__(self):
        # Set up the world
        self.world = LisaWorld("Lisa's World")

        # Create and initialize Lisa
        self.lisa = TinyPerson(name="Lisa")
        self.configure_lisa()

        # Set up knowledge base and load into semantic memory
        knowledge_dir = setup_lisa_knowledge()
        self.lisa.read_documents_from_folder(knowledge_dir)

        # Add Lisa to the world
        self.world.add_agent(self.lisa)

    def configure_lisa(self):
        """Configure Lisa's traits and faculties"""
        # Basic traits
        self.lisa.define("age", "32")
        self.lisa.define("nationality", "American")
        self.lisa.define("occupation", "Data Scientist")
        self.lisa.define("country_of_residence", "United States")

        # Add personality traits
        self.lisa.define_several("personality_traits", [
            "analytical",
            "creative",
            "empathetic",
            "curious about patterns",
            "detail-oriented"
        ])

        # Add professional interests
        self.lisa.define_several("professional_interests", [
            "machine learning",
            "data visualization",
            "ethical AI",
            "statistical modeling"
        ])

        # Add RecallFaculty for semantic memory access
        self.lisa.add_mental_faculties([RecallFaculty()])

    def explore_knowledge(self, topic: str):
        """Have Lisa explore a specific topic using her semantic memory."""
        print(f"\nLisa is exploring knowledge about: {topic}")

        # First, have Lisa think about the topic
        self.lisa.think(f"Let me recall what I know about {topic}...")

        # Use RECALL action to access semantic memory
        self.lisa.listen_and_act(f"RECALL information about {topic}")

        # Process any thoughts or insights
        actions = self.lisa.pop_latest_actions()
        for action in actions:
            if action['type'] == 'THINK':
                print(f"\nLisa's Thoughts: {action['content']}")

    def generate_life_slice(self, topic: Optional[str] = None):
        """Generate a slice of Lisa's life, optionally focused on a specific topic"""
        if topic:
            self.lisa.think(f"Let me recall my experiences with {topic}")
            self.lisa.listen_and_act(f"RECALL experiences and knowledge about {topic}")

        # Create story context
        story = TinyStory(
            agent=self.lisa,
            purpose="Share knowledge and experiences",
            context=f"Reflecting on {'my work and knowledge' if not topic else topic}"
        )

        # Generate narrative
        narrative = story.start_story(
            requirements="Share technical insights and personal experiences",
            number_of_words=300
        )

        return narrative


def main():
    """Main execution function for Lisa's life simulation"""
    simulator = LisaLifeSimulator()

    print("\n=== Initializing Lisa's Life Simulator ===")

    # Explore different areas of knowledge
    topics = [
        "machine learning algorithms",
        "data visualization techniques",
        "project management experiences",
        "technical mentoring",
        "balancing creativity and analysis"
    ]

    for topic in topics:
        print(f"\n=== Exploring {topic} ===")
        simulator.explore_knowledge(topic)

        print("\n=== Generating Related Experience ===")
        narrative = simulator.generate_life_slice(topic)
        print(narrative)


if __name__ == "__main__":
    main()