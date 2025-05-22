from openai import OpenAI
import os
import json
from datetime import datetime
import uuid
import time

# -------------------- Configuration --------------------

# Set your OpenAI API key here or use environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')

# Base directory for storing event documentation
BASE_DIR = 'Event_Documentations'

# Ensure the base directory exists
os.makedirs(BASE_DIR, exist_ok=True)

# File names for metadata and progress tracking
METADATA_FILENAME = 'metadata.json'
SESSION_LOG_FILENAME = 'session_log.txt'
CURRENT_PROGRESS_FILENAME = 'current_progress.json'

# Delay between API calls to respect rate limits
API_CALL_DELAY = 1  # seconds

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------- Helper Functions --------------------

def generate_unique_id():
    """Generates a unique identifier using UUID4."""
    return str(uuid.uuid4())


def get_current_timestamp():
    """Returns the current timestamp as a string."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def save_json(data, filepath):
    """Saves a dictionary as a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def load_json(filepath):
    """Loads a JSON file and returns a dictionary."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}


def append_to_log(folder_path, message):
    """Appends a message to the session log."""
    try:
        log_path = os.path.join(folder_path, SESSION_LOG_FILENAME)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        return True
    except Exception as e:
        print(f"Error writing to log: {e}")
        return False


def create_folder(unique_id):
    """Creates a unique folder for the event documentation."""
    timestamp = get_current_timestamp()
    folder_name = f"Event_Documentation_{unique_id}_{timestamp}"
    folder_path = os.path.join(BASE_DIR, folder_name)
    try:
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    except Exception as e:
        print(f"Error creating folder: {e}")
        return None


def pause():
    """Pauses the script for a short duration to respect API rate limits."""
    time.sleep(API_CALL_DELAY)


def ask_gpt(prompt, max_tokens=150, temperature=0.7):
    """Sends a prompt to OpenAI's GPT-4 and returns the response."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Fixed model name
            messages=[
                {"role": "system",
                 "content": "You are an assistant that helps in documenting events and creating timelines."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return ""


# -------------------- Main Workflow Functions --------------------

def gather_user_input():
    """Step 1: Gather User Input."""
    print("\nStep 1: Gather User Input")
    print("Please describe the event, scenario, or idea you want to document.")
    print("Share as much detail as possible in your initial input.")
    return input("Your Description: ").strip()


def generate_follow_up_questions(initial_input):
    """Step 2: Generate Follow-Up Questions."""
    print("\nStep 2: Generate Follow-Up Questions")
    prompt = (
        f"Based on the following event description, generate three context-relevant "
        f"follow-up questions to clarify and expand on the input:\n\n{initial_input}\n\n"
        f"Provide only the questions numbered 1, 2, and 3."
    )
    questions = ask_gpt(prompt, max_tokens=300)
    pause()
    print("\nGenerated Follow-Up Questions:")
    print(questions)
    return questions


def collect_follow_up_responses(questions):
    """Collects user responses to the follow-up questions."""
    print("\nPlease answer the following follow-up questions:")
    responses = {}
    for line in questions.split('\n'):
        if line.strip() and '.' in line:
            num, question = line.split('.', 1)
            num = num.strip()
            question = question.strip()
            answer = input(f"\n{num}. {question}\nYour Answer: ").strip()
            responses[num] = {"question": question, "answer": answer}
    return responses


def extract_keywords(text):
    """Extracts keywords from the initial input using GPT-4."""
    prompt = (
        f"Extract a list of relevant keywords or tags from the following text:\n\n"
        f"{text}\n\nProvide the keywords as a comma-separated list."
    )
    keywords = ask_gpt(prompt, max_tokens=60)
    pause()
    return [keyword.strip() for keyword in keywords.split(',')]


def generate_metadata(unique_id, initial_input, responses):
    """Generates metadata for the event documentation."""
    return {
        "unique_id": unique_id,
        "creation_date": get_current_timestamp(),
        "initial_input": initial_input,
        "follow_up_responses": responses,
        "tags": extract_keywords(initial_input)
    }


def generate_initial_event_description(folder_path, initial_input, responses):
    """Step 4: Generate Initial Event Description."""
    print("\nStep 4: Generate Initial Event Description")
    prompt = (
        f"Create a detailed and structured description of the event based on the following information.\n\n"
        f"Initial Description:\n{initial_input}\n\n"
        f"Follow-Up Responses:\n"
    )
    for num, qa in responses.items():
        prompt += f"{num}. {qa['question']}\nAnswer: {qa['answer']}\n"

    prompt += (
        "\nInclude the following sections with clear headers:\n"
        "I. Background and Context\n"
        "II. Key Events or Turning Points\n"
        "III. Stakeholders or Participants\n"
        "IV. Implications and Outcomes\n"
    )

    description = ask_gpt(prompt, max_tokens=1000)
    pause()

    description_path = os.path.join(folder_path, 'detailed_event_description.txt')
    try:
        with open(description_path, 'w', encoding='utf-8') as f:
            f.write(description)
        append_to_log(folder_path, "Generated detailed_event_description.txt")
        return description
    except Exception as e:
        print(f"Error saving description: {e}")
        return None


def generate_timeline_and_documents(folder_path, initial_input, responses):
    """Step 5: Generate Timeline and Document Structure."""
    print("\nStep 5: Generate Timeline and Document Structure")
    prompt = (
        "Based on the following event description and follow-up responses, create a JSON structure containing:\n"
        "{\n"
        '  "Before": [\n'
        "    {\n"
        '      "Event Title": "string",\n'
        '      "Event Description": "string",\n'
        '      "Order": "number",\n'
        '      "Documents": [\n'
        "        {\n"
        '          "Title": "string",\n'
        '          "Author": "string",\n'
        '          "Type": "string",\n'
        '          "Description": "string",\n'
        '          "Length": "short|medium|long",\n'
        '          "Status": "Yes|No"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "During": [...],\n'
        '  "After": [...]\n'
        "}\n\n"
        f"Event Description:\n{initial_input}\n\n"
        "Follow-Up Responses:\n"
    )

    for num, qa in responses.items():
        prompt += f"{num}. {qa['question']}\nAnswer: {qa['answer']}\n"

    timeline_with_docs = ask_gpt(prompt, max_tokens=1500)
    pause()

    timeline_path = os.path.join(folder_path, 'timeline_with_document_list.txt')
    try:
        with open(timeline_path, 'w', encoding='utf-8') as f:
            f.write(timeline_with_docs)
        append_to_log(folder_path, "Generated timeline_with_document_list.txt")
        return timeline_with_docs
    except Exception as e:
        print(f"Error saving timeline: {e}")
        return None


def parse_timeline(timeline_text):
    """Parses the timeline text into a Python dictionary."""
    try:
        return json.loads(timeline_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing timeline JSON: {e}")
        return {"Before": [], "During": [], "After": []}


def generate_documents(folder_path, timeline_data):
    """Step 6: Generate Each Document."""
    print("\nStep 6: Generate Each Document")
    documents = []

    for phase in ["Before", "During", "After"]:
        for event in timeline_data.get(phase, []):
            event_title = event.get('Event Title', 'Untitled Event')
            for doc in event.get('Documents', []):
                try:
                    doc_title = doc.get('Title', 'Untitled Document')
                    doc_author = doc.get('Author', 'Author Unknown')
                    doc_type = doc.get('Type', 'report')
                    doc_description = doc.get('Description', '')
                    doc_length = doc.get('Length', 'medium')

                    content_prompt = (
                        f"Create a detailed {doc_type} titled '{doc_title}' authored by '{doc_author}'.\n\n"
                        f"Description: {doc_description}\n\n"
                        f"Content Length: {doc_length.capitalize()}.\n\n"
                        f"Structure the document with appropriate sections."
                    )

                    document_content = ask_gpt(content_prompt, max_tokens=2000)
                    pause()

                    safe_title = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in doc_title)
                    safe_title = safe_title.replace(' ', '_')
                    doc_filename = f"document_{safe_title}_{uuid.uuid4().hex[:8]}.txt"
                    doc_path = os.path.join(folder_path, doc_filename)

                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(document_content)

                    doc['Status'] = 'Yes'
                    documents.append(doc)
                    append_to_log(folder_path, f"Generated {doc_filename}")

                except Exception as e:
                    print(f"Error generating document {doc_title}: {e}")
                    continue

    # Update timeline file with new statuses
    timeline_path = os.path.join(folder_path, 'timeline_with_document_list.txt')
    try:
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=4)
        append_to_log(folder_path, "Updated timeline_with_document_list.txt with document statuses")
    except Exception as e:
        print(f"Error updating timeline file: {e}")

    return documents


def save_progress(folder_path, current_step, data):
    """Saves the current progress to a JSON file."""
    progress = {
        "current_step": current_step,
        "data": data,
        "timestamp": get_current_timestamp()
    }
    progress_path = os.path.join(folder_path, CURRENT_PROGRESS_FILENAME)
    return save_json(progress, progress_path)


def resume_session(folder_path, progress_data):
    """Resumes a session from the saved progress."""
    current_step = progress_data.get("current_step", "")
    data = progress_data.get("data", {})

    print(f"\nResuming from: {current_step}")

    if current_step == "Step 4: Generated Initial Event Description":
        timeline_text = generate_timeline_and_documents(folder_path, data.get("initial_input", ""),
                                                        data.get("responses", {}))
        timeline_data = parse_timeline(timeline_text)
        documents = generate_documents(folder_path, timeline_data)

    elif current_step == "Step 6: Generating Documents":
        timeline_data = data.get("timeline_data", {})
        documents = generate_documents(folder_path, timeline_data)

    else:
        print("Unable to determine resume point. Starting new session.")
        return False

    return True


def main():
    """Main function to execute the workflow."""
    print("=== Enhanced Structured Prompt for Event Documentation ===\n")

    # Check for resume
    resume = input("Do you want to resume a previous session? (yes/no): ").strip().lower()

    if resume == 'yes':
        folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
        if not folders:
            print("No existing sessions found. Starting a new session.")
            resume = 'no'
        else:
            print("\nAvailable Sessions:")
            for idx, folder in enumerate(folders, 1):
                print(f"{idx}. {folder}")

            try:
                choice = int(input("\nSelect a session number to resume: "))
                if 1 <= choice <= len(folders):
                    selected_folder = os.path.join(BASE_DIR, folders[choice - 1])
                    progress = load_json(os.path.join(selected_folder, CURRENT_PROGRESS_FILENAME))

                    if progress:
                        if resume_session(selected_folder, progress):
                            print("\nSession resumed and completed successfully!")
                            return
                    else:
                        print("No progress file found. Starting new session.")
                        resume = 'no'
                else:
                    print("Invalid selection. Starting new session.")
                    resume = 'no'
            except ValueError:
                print("Invalid input. Starting new session.")
                resume = 'no'

    if resume == 'no':
        try:
            # Start new session
            unique_id = generate_unique_id()
            folder_path = create_folder(unique_id)

            if not folder_path:
                print("Error creating documentation folder. Exiting.")
                return

            append_to_log(folder_path, "Started new event documentation session.")

            # Step 1: Gather User Input
            initial_input = gather_user_input()
            if not initial_input:
                print("No input provided. Exiting.")
                return
            append_to_log(folder_path, "Collected initial user input.")

            # Step 2: Generate Follow-Up Questions
            questions = generate_follow_up_questions(initial_input)
            if not questions:
                print("Error generating follow-up questions. Exiting.")
                return

            # Step 3: Collect Follow-Up Responses
            responses = collect_follow_up_responses(questions)
            if not responses:
                print("No follow-up responses collected. Exiting.")
                return
            append_to_log(folder_path, "Collected follow-up responses.")

            # Step 4: Generate Metadata
            metadata = generate_metadata(unique_id, initial_input, responses)
            metadata_path = os.path.join(folder_path, METADATA_FILENAME)
            if not save_json(metadata, metadata_path):
                print("Error saving metadata. Continuing anyway.")
            append_to_log(folder_path, f"Saved metadata to {METADATA_FILENAME}.")

            # Save progress after initial steps
            save_progress(folder_path, "Step 4: Generated Initial Event Description", {
                "folder_path": folder_path,
                "initial_input": initial_input,
                "responses": responses
            })

            # Step 4: Generate Initial Event Description
            description = generate_initial_event_description(folder_path, initial_input, responses)
            if not description:
                print("Error generating event description. Exiting.")
                return

            # Step 5: Generate Timeline and Document Structure
            timeline_text = generate_timeline_and_documents(folder_path, initial_input, responses)
            if not timeline_text:
                print("Error generating timeline. Exiting.")
                return

            timeline_data = parse_timeline(timeline_text)

            # Save progress before generating documents
            save_progress(folder_path, "Step 6: Generating Documents", {
                "folder_path": folder_path,
                "timeline_data": timeline_data
            })

            # Step 6: Generate Each Document
            documents = generate_documents(folder_path, timeline_data)

            if documents:
                print("\nAll documents have been generated successfully!")
                print(f"\nDocumentation folder: {folder_path}")
                print("\nGenerated files:")
                print(f"- {METADATA_FILENAME}")
                print(f"- {SESSION_LOG_FILENAME}")
                print("- detailed_event_description.txt")
                print("- timeline_with_document_list.txt")
                print(f"- {len(documents)} additional document files")

                append_to_log(folder_path, "All documents generated successfully.")
            else:
                print("\nWarning: No documents were generated.")
                append_to_log(folder_path, "Warning: Document generation completed with no documents.")

        except Exception as e:
            print(f"\nAn error occurred during execution: {e}")
            if 'folder_path' in locals():
                append_to_log(folder_path, f"Error during execution: {e}")
            return

        print("\nProcess completed. All files are saved in the respective event documentation folder.")
        print(f"Folder Path: {folder_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\nExiting program.")