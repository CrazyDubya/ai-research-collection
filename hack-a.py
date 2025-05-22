"""
Hackathon simulation with specialized web development agents.
"""

import textwrap
import json
import logging
from datetime import datetime
import os
from pathlib import Path

from tinytroupe.tools import TinyTool
from tinytroupe.agent import TinyPerson, TinyToolUse
from tinytroupe.environment import TinyWorld
from tinytroupe.extraction import ArtifactExporter, default_extractor
import tinytroupe.utils as utils

logger = logging.getLogger("tinytroupe")

def create_frontend_dev():
    """Creates a frontend developer agent."""
    agent = TinyPerson("Alex")

    agent.define("age", 28)
    agent.define("nationality", "American")
    agent.define("occupation", "Frontend Developer")
    agent.define("occupation_description", textwrap.dedent("""
        Senior frontend developer specializing in modern web frameworks and UI/UX.
        Expert in HTML5, CSS3, JavaScript, and responsive design. Currently leads
        the frontend team at a tech startup working on innovative web applications.
    """))

    agent.define_several("personality_traits", [
        {"trait": "Detail-oriented perfectionist when it comes to UI"},
        {"trait": "Early adopter of new web technologies"},
        {"trait": "Collaborative and documentation-focused"}
    ])

    agent.define_several("professional_interests", [
        {"interest": "Modern JavaScript frameworks"},
        {"interest": "Web accessibility standards"},
        {"interest": "UI/UX best practices"},
        {"interest": "Performance optimization"}
    ])

    agent.define_several("skills", [
        {"skill": "Advanced HTML5 and CSS3"},
        {"skill": "JavaScript/TypeScript expertise"},
        {"skill": "Web animation and transitions"},
        {"skill": "Responsive design"},
        {"skill": "Cross-browser compatibility"}
    ])

    return agent

def create_ui_designer():
    """Creates a UI designer agent."""
    agent = TinyPerson("Maya")

    agent.define("age", 31)
    agent.define("nationality", "Canadian")
    agent.define("occupation", "UI Designer")
    agent.define("occupation_description", textwrap.dedent("""
        Creative UI designer with strong focus on user experience and accessibility.
        Specializes in creating visually appealing and highly usable interfaces.
        Experienced in design systems and component libraries.
    """))

    agent.define_several("personality_traits", [
        {"trait": "Creative and innovative in design approaches"},
        {"trait": "Strong advocate for user-centered design"},
        {"trait": "Analytical about user behavior"}
    ])

    agent.define_several("professional_interests", [
        {"interest": "Design systems"},
        {"interest": "Motion design"},
        {"interest": "Color theory"},
        {"interest": "Typography"}
    ])

    agent.define_several("skills", [
        {"skill": "UI/UX design principles"},
        {"skill": "CSS animations"},
        {"skill": "Design tools mastery"},
        {"skill": "Prototyping"},
        {"skill": "Accessibility guidelines"}
    ])

    return agent

def create_data_viz_dev():
    """Creates a data visualization developer agent."""
    agent = TinyPerson("Raj")

    agent.define("age", 29)
    agent.define("nationality", "Indian")
    agent.define("occupation", "Data Visualization Developer")
    agent.define("occupation_description", textwrap.dedent("""
        Specialized in creating interactive data visualizations for web applications.
        Expert in D3.js, Chart.js, and other visualization libraries. Strong background
        in both data analysis and frontend development.
    """))

    agent.define_several("personality_traits", [
        {"trait": "Analytical and detail-oriented"},
        {"trait": "Passionate about data storytelling"},
        {"trait": "Innovation-focused"}
    ])

    agent.define_several("professional_interests", [
        {"interest": "Data visualization techniques"},
        {"interest": "Interactive graphics"},
        {"interest": "Statistical analysis"},
        {"interest": "Performance optimization"}
    ])

    agent.define_several("skills", [
        {"skill": "D3.js expertise"},
        {"skill": "SVG animation"},
        {"skill": "Data processing"},
        {"skill": "JavaScript performance"},
        {"skill": "Responsive visualizations"}
    ])

    return agent

def setup_agent_cognitive_state(agent):
    """Sets up complete cognitive state for an agent with defaults."""
    cognitive_defaults = {
        "goals": ["Write clean code", "Follow best practices"],
        "attention": "Code implementation",
        "emotions": "Focused and ready",
        "context": ["hackathon", "web development"],
        "mental_state": "Ready to work",
        "motivation": "High",
        "knowledge_state": "Active",
        "collaboration_state": "Open to teamwork",
        "current_task": "Not assigned",
        "progress": "Not started",
        "mood": "Professional",
        "fatigue": "Fresh",
        "creativity": "High"
    }

    for field, value in cognitive_defaults.items():
        if agent.get(field) is None:
            agent.define(field, value)

def run_hackathon(output_dir="./hackathon_output"):
    """Run a web development hackathon simulation."""

    os.makedirs(output_dir, exist_ok=True)
    exporter = ArtifactExporter(base_output_folder=output_dir)

    agents = [
        create_frontend_dev(),
        create_ui_designer(),
        create_data_viz_dev()
    ]

    for agent in agents:
        setup_agent_cognitive_state(agent)

    hackathon_team = TinyWorld(
        "Web Development Hackathon",
        agents=agents,
        initial_datetime=datetime.now()
    )

    # Hackathon simulation logic
    hackathon_team.broadcast(
        "Welcome to the hackathon! Start by creating basic structure for your tasks."
    )

    results = hackathon_team.run(steps=10, return_actions=True)
    final_output = default_extractor.extract_results_from_world(
        hackathon_team,
        extraction_objective="Retrieve all generated files.",
    )
    return final_output

if __name__ == "__main__":
    results = run_hackathon()
    print("Hackathon simulation completed!")
