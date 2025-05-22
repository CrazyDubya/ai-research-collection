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
word_processor = TinyWordProcessor(exporter=exporter, enricher=enricher)  # Define the word processor
tooluse_faculty = TinyToolUse(tools=[word_processor])  # Pass the word processor to the faculty

instructions = f"""Below is a structured prompt that includes all the actions defined in the provided code, along with their descriptions, constraints, and examples.

```plaintext
# Action Definitions and Constraints

## 1. RECALL
- **Description**: You can recall information from your memory. To do so, you must specify a "mental query" to locate the desired memory. If the memory is found, it is brought to your conscience.
- **Constraints**:
  - Before concluding you don't know something or don't have access to some information, you **must** try to RECALL it from your memory.
  - You try to RECALL information from your semantic/factual memory, so that you can have more relevant elements to think and talk about, whenever such an action would be likely to enrich the current interaction.
  - If you RECALL, you use a "mental query" that describes the elements you are looking for, not a question. It is like a keyword-based search query.
  - It may take several tries of RECALL to get the relevant information you need. Be creative with your queries.
- **Example**:
<THINK something> <RECALL "cat products"> <THINK something> <RECALL "feline artifacts"> <THINK something> <RECALL "pet store"> <THINK something> <TALK something> DONE ```
2. CONSULT

Description: You can retrieve and consult a specific document, so that you can access its content and accomplish your goals. To do so, you specify the name of the document you want to consult.
Constraints:
If you need information about a specific document, you must use CONSULT instead of RECALL.
You can only CONSULT a few documents before issuing DONE.
Example:
LIST_DOCUMENTS
<CONSULT "some document name">
<THINK something about the retrieved document>
<TALK something>
DONE
3. LIST_DOCUMENTS

Description: You can list the documents you have access to, so that you can decide which to access, if any, to accomplish your goals. Documents include any kind of "packaged" information you can access, such as emails, files, chat messages, calendar events, etc.
Constraints:
You must use LIST_DOCUMENTS before CONSULTING a document to know which documents are available.
Example:
LIST_DOCUMENTS
<CONSULT "some document name">
<THINK something about the retrieved document>
<TALK something>
DONE
4. THINK

Description: You can think about something and update your internal cognitive state.
Constraints:
You can interleave THINK and RECALL to better reflect on the information you are trying to recall.
Example:
<THINK something>
<RECALL "cat products">
<THINK something>
<TALK something>
DONE
5. TALK

Description: You can talk to another agent or the environment, producing a conversation.
Constraints:
You must specify the target of your conversation.
Example:
<TALK to "Agent1" about "cat products">
DONE
6. DONE

Description: You conclude your sequence of actions.
Constraints:
You must issue DONE after a sequence of actions.
Example:
<THINK something>
<RECALL "cat products">
<THINK something>
<TALK something>
DONE
Usage Instructions

RECALL: Use this action to retrieve information from your memory. Specify a "mental query" to locate the desired memory.
CONSULT: Use this action to access a specific document by its name.
LIST_DOCUMENTS: Use this action to list all available documents before consulting them.
THINK: Use this action to think about something and update your internal cognitive state.
TALK: Use this action to engage in a conversation with another agent or the environment.
DONE: Use this action to conclude your sequence of actions.
Examples

Example 1: Retrieving information about cats.
<THINK something>
<RECALL "cat products">
<THINK something>
<RECALL "feline artifacts">
<THINK something>
<RECALL "pet store">
<THINK something>
<TALK something>
DONE
Example 2: Consulting a specific document.
LIST_DOCUMENTS
<CONSULT "Cat Care Guide.pdf">
<THINK something about the retrieved document>
<TALK something>
DONE
Example 3: Engaging in a conversation.
<TALK to "Agent1" about "cat products">
DONE

**Usage Instructions Summary:**
- **RECALL**: Retrieve information from memory using a "mental query."
- **CONSULT**: Access a specific document by name.
- **LIST_DOCUMENTS**: List all available documents before consulting.
- **THINK**: Reflect and update internal cognitive state.
- **TALK**: Engage in conversation with another agent or environment.
- **DONE**: Conclude the sequence of actions.

By providing this structured prompt to the LLM, you ensure that it understands the actions available to the agent, how to use them, and the constraints that apply to each action. This will help the LLM generate coherent and contextually appropriate responses.
"""

# Create agents
lisa1 = create_lisa_the_data_scientist()
oscar = create_oscar_the_architect()
marcos = create_marcos_the_physician()  # Fixed variable name from "alex" to "marcos"

# Set up Lisa's mental faculties
lisa1.add_mental_faculties([tooluse_faculty])
oscar.add_mental_faculties([tooluse_faculty])
marcos.add_mental_faculties([RecallFaculty()])  # Marcos only needs recall faculty

# Create the world and add agents
world = TinyWorld("Wordprocessor Tool Usage", agents=[lisa1, oscar, marcos])

# Function to get user input for the story concept
def get_user_concept():
    concept = input("Enter the story concept: ")
    return concept

    lisa1.listen(instructions)
    lisa1.listen_and_act(f"Write outline for story and save it as outline, based on the concept: {concept}")
    lisa1.listen_and_act(f"List and load document outline and expand the setting")
    lisa1.listen_and_act(f"List and load document outline and expand the characters")
    lisa1.listen_and_act(f"List and load document outline and expand the history")
    lisa1.listen_and_act(f"List and load document outline and develop a subplot")
    lisa1.listen_and_act(f"List and load document outline and develop/expand the climax")
    lisa1.listen_and_act(f"List and load document outline and develop/expand the resolution")
    lisa1.listen_and_act(f"List and load document outline and develop/expand the ending")


# Recursive function to build the story
def build_story(agent, concept, chapter=1):
    if chapter > 10:
        print("Story completed!")
        return

    # Lisa writes the outline or chapter content
    lisa1.listen(instructions)
    lisa1.listen_and_act(f"Write outline for story and save it as outline,  based on the concept: {concept}")
    lisa1.listen(f"Write chapter {chapter} based on the concept: {concept} and the outline file. Be witty, be experienced, avoid the purple and cliche.")
    lisa1.act(until_done=True)
    lisa1.listen_and_act(f"List document to find chapter {chapter} example [Chapter 2.Lisa.md] and expand the setting")
    lisa1.listen_and_act(f"List document to find chapter {chapter} and expand the characters")
    lisa1.listen_and_act(f"List document to find chapter {chapter} and improve the prose")
    lisa1.listen_and_act(f"List document to find chapter {chapter} and expand the plot")
    lisa1.listen_and_act(f"List document to find chapter {chapter} and improve the prose")
    lisa1.act(until_done=True)
    actions = lisa1.pop_latest_actions()
    for action in actions:
        if action['type'] == 'WRITE_DOCUMENT':
            document_content = action['content']
            word_processor.write_document(title=f"Chapter {chapter}", content=document_content, author=lisa1.name)

    # Oscar edits the content
    oscar.listen(instructions)
    oscar.listen(f"Edit chapter {chapter}. Give feedbakc to Lisa. Don't forget to list documents to find it")
    oscar.act(until_done=True)
    lisa1.listen_and_act(f"List document to find chapter {chapter} and revise based on Oscar feedback")

    # Marcos reads and provides feedback
    marcos.listen(instructions)
    marcos.listen(f"Read and provide feedback on chapter {chapter} to Lisa.  Don't forget to list documents to find it")
    marcos.act(until_done=True)
    world.run(2)
    lisa1.listen_and_act(f"List document to find chapter {chapter} and revise based on Marcos feedback")
    lisa1.listen(instructions)
    lisa1.act(until_done=True)
    # Recursive call for the next chapter
    build_story(agent, concept, chapter + 1)

# Main process
concept = get_user_concept()
build_story(lisa1, concept)