# Wordprocessor Usage Example


import json
import sys
import csv
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
from examples import create_lisa_the_data_scientist, create_oscar_the_architect, create_marcos_the_physician

# Set up export folder
data_export_folder = "../data/extractions/wordprocessor"

# Initialize components
exporter = ArtifactExporter(base_output_folder=data_export_folder)
enricher = TinyEnricher()
tooluse_faculty = TinyToolUse(tools=[TinyWordProcessor(exporter=exporter, enricher=enricher)])

# Create agents
lisa = create_lisa_the_data_scientist()
oscar = create_oscar_the_architect()
alex = create_marcos_the_physician()

# Set up Lisa's mental faculties
LisaRecallFaculty = RecallFaculty()  # Lisa's recall faculty
lisasemanticmemory = SemanticMemory()  # Lisa's semantic memory
lisa.add_mental_faculties([tooluse_faculty])
oscar.add_mental_faculties([tooluse_faculty])
world = TinyWorld("Wordprocessor Tool Usage", agents=[lisa, oscar, alex])

# Agent interactions
lisa.listen_and_act("""
Please use the WRITE_DOCUMENT tool with this exact JSON structure:
{
    "type": "WRITE_DOCUMENT",
    "content": {
        "title": "[Choose a title]",
        "content": "[Your detailed response in Markdown format]",
        "author": "Lisa the [Choose a fun nickname]"
    }
}
Write outline for a fictional soapy whodunnit involving a city school PTA, a bake sale, an affair, and a 40 year old secret""")

oscar.act("""
{
    "type": "WRITE_DOCUMENT",
    "content": {
        "title": "The Architect's Plan",
        "content": "The architect's plan was to build a city school PTA, a bake sale, an affair, and a 40 year old secret into a fictional soapy whodunnit.",
        "author": "Oscar the Architect"
    }
}
""")

# Continue agent interactions
lisa.think_and_act("I've written the outline for a fictional soapy whodunnit involving a city school PTA, a bake sale, an affair, and a 40 year old secret.")
lisa_speech = ("I've written the outline for a fictional soapy whodunnit involving a city school PTA, a bake sale, an affair, and a 40 year old secret.")
oscar.listen(lisa_speech)

# Revision cycle
for _ in range(3):
    lisa.listen_and_act("Revise your essay with a new title and new spin.")
    world.run_months(5)
    lisa.listen_and_act("Please use the WRITE_DOCUMENT tool and revise again with new title")
    world.run_months(12)

# Final revisions
lisa.listen_and_act("""
Please use the WRITE_DOCUMENT tool with this exact JSON structure:
{
    "type": "WRITE_DOCUMENT",
    "content": {
        "title": "[Choose a title]",
        "content": "[Your detailed response in Markdown format]",
        "author": "Lisa the [Choose a fun nickname]"
    }
}
Write revised outline for a fictional soapy whodunnit involving a city school PTA, a bake sale, an affair, and a 40 year old secret""")

lisa.listen_and_act("""
Please use the WRITE_DOCUMENT tool with this exact JSON structure:
{
    "type": "WRITE_DOCUMENT",
    "content": {
        "title": "[Choose a Chapter title]",
        "content": "[Your detailed response in Markdown format]",
        "author": "Lisa the [Choose a fun nickname]"
    }
}
Write Chapter 1 for your story in prose. avoid the purple. be detailed and descriptive. use the tools of the trade. """)
