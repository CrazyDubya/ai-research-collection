"""
Hackathon simulation with specialized web development agents.
"""

import textwrap
import json
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path

import pandas as pd

from tinytroupe.tools import TinyTool
from tinytroupe.agent import TinyPerson, TinyToolUse
from tinytroupe.environment import TinyWorld
from tinytroupe.extraction import ArtifactExporter, default_extractor
from tinytroupe.enrichment import TinyEnricher
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


"""
Fixed hackathon implementation with proper agent actions.
"""

"""
Updated hackathon implementation with comprehensive cognitive state handling.
"""

"""
Enhanced hackathon implementation that encourages active coding.
"""


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

                utils.check_valid_fields(spec, ["filename", "code", "language"])

                self.files[spec['filename']] = {
                    'code': spec['code'],
                    'language': spec['language']
                }

                agent.think(
                    f"I've written code for {spec['filename']}. Let me review it and make any needed improvements.")

                if self.exporter:
                    self.exporter.export(
                        artifact_name=spec['filename'],
                        artifact_data=spec['code'],
                        content_type=spec['language'],
                        target_format=spec['language'].lower()
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

        Example WRITE_CODE action:
        {
            "type": "WRITE_CODE",
            "content": {
                "filename": "index.html",
                "language": "html",
                "code": "<!DOCTYPE html><html><head><title>My Page</title></head><body><h1>Hello</h1></body></html>"
            }
        }

        You should write code frequently! Start with basic structure, then improve iteratively.
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


"""
Enhanced hackathon implementation that encourages active coding.
"""

"""
Enhanced hackathon implementation that encourages active coding.
"""


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
        try:
            current = agent.get(field)
            if current is None:
                agent.define(field, value)
        except:
            agent.define(field, value)


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

                utils.check_valid_fields(spec, ["filename", "code", "language"])

                # Normalize newlines and clean up code
                code = spec['code'].replace('\\n', '\n').strip()

                self.files[spec['filename']] = {
                    'code': code,
                    'language': spec['language']
                }

                agent.think(
                    f"I've written code for {spec['filename']}. Let me review it and make any needed improvements.")

                if self.exporter:
                    # Map language to proper target format
                    format_map = {
                        'html': 'txt',
                        'css': 'txt',
                        'javascript': 'txt'
                    }
                    target_format = format_map.get(spec['language'].lower(), 'txt')

                    self.exporter.export(
                        artifact_name=spec['filename'],
                        artifact_data=code,
                        content_type=spec['language'],
                        target_format=target_format
                    )
                return True

            except json.JSONDecodeError as e:
                logger.error(f"JSON Error: {str(e)}")
                agent.think(
                    f"I need to make sure my code is proper JSON with escaped quotes and newlines. Let me try again.")
                return False
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

        Example WRITE_CODE actions:

        For HTML:
        {
            "type": "WRITE_CODE",
            "content": {
                "filename": "form.html",
                "language": "html",
                "code": "<!DOCTYPE html>\\n<html>\\n<head>\\n    <title>Form</title>\\n</head>\\n<body>\\n    <h1>Form</h1>\\n    <form>\\n        <input type=\\"text\\">\\n    </form>\\n</body>\\n</html>"
            }
        }

        For CSS:
        {
            "type": "WRITE_CODE", 
            "content": {
                "filename": "styles.css",
                "language": "css",
                "code": "body {\\n    margin: 0;\\n    padding: 20px;\\n}\\n\\nform {\\n    max-width: 500px;\\n}"
            }
        }

        For JavaScript:
        {
            "type": "WRITE_CODE",
            "content": {
                "filename": "script.js",
                "language": "javascript", 
                "code": "document.addEventListener(\\"DOMContentLoaded\\", () => {\\n    const form = document.querySelector(\\"form\\");\\n});"
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


def run_hackathon(output_dir: str = "./hackathon_output"):
    """Run a web development hackathon simulation."""

    os.makedirs(output_dir, exist_ok=True)

    exporter = ArtifactExporter(base_output_folder=output_dir)
    code_editor = HackathonCodeEditor(exporter=exporter)

    agents = []
    for create_fn in [create_frontend_dev, create_ui_designer, create_data_viz_dev]:
        agent = create_fn()
        agent.add_mental_faculties([TinyToolUse([code_editor])])
        setup_agent_cognitive_state(agent)
        agents.append(agent)

    coding_team = TinyWorld(
        "Web Dev Hackathon",
        agents=agents,
        initial_datetime=datetime.now()
    )
    coding_team.make_everyone_accessible()

    # Give more specific coding instructions with examples
    initial_tasks = {
        "frontend": """
        Create form.html and responsive.html:
        1. Start with a basic form structure:
           {
               "type": "WRITE_CODE",
               "content": {
                   "filename": "form.html",
                   "language": "html",
                   "code": "<!DOCTYPE html><html>...</html>"
               }
           }
        2. Add validation using HTML5 and JavaScript
        3. Use CSS Grid/Flexbox for responsive layout
        """,

        "design": """
        Create animations.html and a11y.html:
        1. Begin with animation examples:
           {
               "type": "WRITE_CODE",
               "content": {
                   "filename": "animations.html",
                   "language": "html",
                   "code": "<!DOCTYPE html><html>...</html>"
               }
           }
        2. Add CSS transitions and keyframes
        3. Ensure WCAG compliance for accessibility
        """,

        "dataviz": """
        Create visualization.html:
        1. Start with Chart.js integration:
           {
               "type": "WRITE_CODE",
               "content": {
                   "filename": "visualization.html",
                   "language": "html",
                   "code": "<!DOCTYPE html><html>...</html>"
               }
           }
        2. Add interactive data visualization
        3. Make charts responsive
        """
    }

    # Give each agent their specific task with example code
    for agent, task in zip(agents, initial_tasks.values()):
        agent.listen(task)
        agent.think("I should start writing code right away following these examples")
        agent.internalize_goal("Write code in small, manageable steps")

    # Run simulation with frequent code writing prompts
    for step in range(15):
        coding_team.broadcast("Remember to write code regularly! Use WRITE_CODE to create or update files.")
        results = coding_team.run(steps=1, return_actions=True)

        # Check if any code was written this step
        code_written = False
        for agent_actions in results.values():
            for action in agent_actions:
                if action.get('type') == 'WRITE_CODE':
                    code_written = True
                    break

        # Encourage coding if none happened
        if not code_written:
            coding_team.broadcast("Let's focus on writing some code now. Start with a basic HTML structure!")

    # Extract final HTML pages
    fields = {
        "data_viz_page": "Chart.js visualization page",
        "animation_page": "CSS animations demo",
        "responsive_page": "Responsive layout example",
        "form_page": "Form validation page",
        "a11y_page": "Accessible design page"
    }

    return default_extractor.extract_results_from_world(
        coding_team,
        extraction_objective="Get the final HTML pages created",
        fields=fields
    )


if __name__ == "__main__":
    results = run_hackathon()
    print("Hackathon simulation completed!")