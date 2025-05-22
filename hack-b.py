"""
Hackathon simulation with specialized web development agents.
"""

import textwrap
import json
import logging
from datetime import datetime
import os

from tinytroupe.tools import TinyTool
from tinytroupe.agent import TinyPerson, TinyToolUse
from tinytroupe.environment import TinyWorld
from tinytroupe.extraction import ArtifactExporter, default_extractor
import tinytroupe.utils as utils

logger = logging.getLogger("tinytroupe")


class HackathonCodeEditor(TinyTool):
    """Tool for writing and managing code during the hackathon."""

    def __init__(self, owner=None, exporter=None):
        name = "code_editor"
        description = "A tool for writing and managing HTML/CSS/JS code"
        super().__init__(name=name, description=description, owner=owner, real_world_side_effects=False,
                         exporter=exporter)
        self.files = {}

    def _process_action(self, agent, action: dict) -> bool:
        if action['type'] == "WRITE_CODE" and action['content'] is not None:
            try:
                spec = json.loads(action['content']) if isinstance(action['content'], str) else action['content']

                # Prefill language with "lol" if not provided
                spec.setdefault('language', 'lol')

                utils.check_valid_fields(spec, ["filename", "code", "language"])

                # Handle optional comments
                comments = spec.get('comments', '')

                # Exporting as .txt for now to avoid unsupported format issues
                filename_with_extension = f"{spec['filename']}.txt"
                self.files[filename_with_extension] = {
                    'code': spec['code'],
                    'language': spec['language'],
                    'comments': comments
                }

                agent.think(
                    f"I've written code for {filename_with_extension}. Let me review it and make any needed improvements."
                )

                if self.exporter:
                    self.exporter.export(
                        artifact_name=filename_with_extension,
                        artifact_data=spec['code'],
                        content_type="text/plain",
                        target_format="txt"
                    )
                return True

            except Exception as e:
                logger.error(f"Error processing code: {str(e)}")
                agent.think(f"I encountered an error while writing code: {str(e)}. Let me try again.")
                return False

        return False

    def actions_definitions_prompt(self) -> str:
        return """
        - WRITE_CODE: Create or update code files. Content must be JSON with:
            * filename: The file to create (e.g., 'index.html')
            * code: Your actual HTML/CSS/JS code
            * language: 'html', 'css', or 'javascript'
            * comments (optional): Any comments about the code

        Example WRITE_CODE actions:

        For HTML:
        {
            "type": "WRITE_CODE",
            "content": {
                "filename": "form.html",
                "language": "html",
                "code": "<!DOCTYPE html>\\n<html>\\n<head>\\n    <title>Form</title>\\n</head>\\n<body>\\n    <h1>Form</h1>\\n    <form>\\n        <input type=\\"text\\">\\n    </form>\\n</body>\\n</html>",
                "comments": "Basic form structure"
            }
        }

        For CSS:
        {
            "type": "WRITE_CODE", 
            "content": {
                "filename": "styles.css",
                "language": "css",
                "code": "body {\\n    margin: 0;\\n    padding: 20px;\\n}\\n\\nform {\\n    max-width: 500px;\\n}",
                "comments": "Basic styles for body and form"
            }
        }

        For JavaScript:
        {
            "type": "WRITE_CODE",
            "content": {
                "filename": "script.js",
                "language": "javascript", 
                "code": "document.addEventListener(\\"DOMContentLoaded\\", () => {\\n    const form = document.querySelector(\\"form\\");\\n});",
                "comments": "Add event listener for DOMContentLoaded"
            }
        }

        Important:
        - Use proper JSON escaping for quotes and newlines
        - Format code for readability with \\n for newlines
        - Always include complete, valid code structure
        """

    def actions_constraints_prompt(self) -> str:
        return """
        - Write code in small, iterative steps
        - After each WRITE_CODE action, THINK about what to improve
        - Always include proper HTML structure with doctype
        - Add descriptive comments to your code
        - Use CDNs for external libraries
        - Make frequent commits to track progress
        """

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


def run_hackathon(output_dir="./hackathon_output2"):
    """Run a web development hackathon simulation."""
    os.makedirs(output_dir, exist_ok=True)
    exporter = ArtifactExporter(base_output_folder=output_dir)
    code_editor = HackathonCodeEditor(exporter=exporter)

    agents = [
        create_frontend_dev(),
        create_ui_designer(),
        create_data_viz_dev()
    ]

    for agent in agents:
        setup_agent_cognitive_state(agent)
        agent.add_mental_faculties([TinyToolUse([code_editor])])

    hackathon_team = TinyWorld(
        "Web Development Hackathon",
        agents=agents,
        initial_datetime=datetime.now()
    )

    hackathon_team.broadcast(
        "Welcome to the hackathon! Use filename, code and language as only fields. Start by creating complete well-commented skelton code. Do not create project plans.  functional code only. Dont forget forward progres"
    )

    results = hackathon_team.run(steps=20, return_actions=True)
    final_output = default_extractor.extract_results_from_world(
        hackathon_team,
        extraction_objective="Retrieve all generated files.",
    )
    return final_output


if __name__ == "__main__":
    results = run_hackathon()
    print("Hackathon simulation completed!")
