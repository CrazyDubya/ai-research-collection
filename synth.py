import os
import json
import time
import textwrap
import csv
from pathlib import Path
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, OpenAI
import anthropic
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
NEBIUS_API_KEY_ENV_VAR = "NEBIUS_API_KEY"
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
OUTPUT_DIR = Path("generated_content")
STATE_FILE = OUTPUT_DIR / "state.json"
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # Seconds
DEFAULT_MAX_TOKENS = 15000
DEFAULT_TEMPERATURE = 0.7
OPENAI_DEPLOYMENT_NAME = "gpt-4"
NEBIUS_MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B-Instruct"
OLLAMA_MODEL_NAME = "llama2"
CLAUDE_MODEL_NAME = "claude-3-sonnet-20240229"

# Type aliases
DocumentType = Dict[str, str]
EpisodeType = Dict[str, Any]
StateType = Dict[str, Any]


class APIKeyError(Exception):
    """Custom exception for API key errors."""
    pass


class APICallError(Exception):
    """Custom exception for API call errors."""
    pass


def load_api_keys() -> tuple[str, str, str]:
    """Load the OpenAI, Nebius, and Anthropic API keys from environment variables."""
    openai_key = os.getenv(OPENAI_API_KEY_ENV_VAR)
    nebius_key = os.getenv(NEBIUS_API_KEY_ENV_VAR)
    anthropic_key = os.getenv(ANTHROPIC_API_KEY_ENV_VAR)

    if not openai_key:
        raise APIKeyError(f"Please set the {OPENAI_API_KEY_ENV_VAR} environment variable.")
    if not nebius_key:
        raise APIKeyError(f"Please set the {NEBIUS_API_KEY_ENV_VAR} environment variable.")
    if not anthropic_key:
        raise APIKeyError(f"Please set the {ANTHROPIC_API_KEY_ENV_VAR} environment variable.")

    return openai_key, nebius_key, anthropic_key


async def call_api(api_choice: str, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                   temperature: float = DEFAULT_TEMPERATURE) -> str:
    """Call the selected API with the given prompt."""
    for attempt in range(MAX_RETRIES):
        try:
            if api_choice == "openai":
                openai_key, _, _ = load_api_keys()
                client = AsyncOpenAI(api_key=openai_key)
                response = await client.chat.completions.create(
                    model=OPENAI_DEPLOYMENT_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            elif api_choice == "nebius":
                _, nebius_key, _ = load_api_keys()
                client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=nebius_key)
                completion = client.chat.completions.create(
                    model=NEBIUS_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content.strip()
            elif api_choice == "ollama":
                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
                completion = client.chat.completions.create(
                    model=OLLAMA_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content.strip()
            elif api_choice == "anthropic":
                _, _, anthropic_key = load_api_keys()
                client = anthropic.Anthropic(api_key=anthropic_key)
                message = client.messages.create(
                    model=CLAUDE_MODEL_NAME,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text.strip()
            else:
                raise ValueError(f"Invalid API choice: {api_choice}")
        except Exception as e:
            logging.error(f"{api_choice.capitalize()} API error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
            else:
                raise APICallError(f"Max retries reached for {api_choice.capitalize()} API call.") from e


def parse_json_response(response: str) -> Any:
    """Parse JSON response with fallback mechanisms."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from the response if it's embedded in other text
        try:
            json_start = response.index('{')
            json_end = response.rindex('}') + 1
            return json.loads(response[json_start:json_end])
        except (ValueError, json.JSONDecodeError):
            # If still fails, return the raw string
            logging.warning("Failed to parse JSON. Returning raw string.")
            return response


def save_state(state: StateType):
    """Save the current state to a file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def load_state() -> Optional[StateType]:
    """Load the state from a file if it exists."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None


async def generate_detailed_world(story_details: str, api_choice: str) -> str:
    """Generate a detailed world based on the user's story description."""
    prompt = f"""
    Expand the following story description into a detailed world, including:
    1. History of the world
    2. Main characters and their backgrounds
    3. Key locations and their significance
    4. Major plot points and story arcs
    5. Unique aspects of the world (e.g., technology, magic systems, cultural practices)

    Story Description:
    {story_details}

    Provide a comprehensive and cohesive world description that can support multiple episodes and diverse document types.
    """
    return await call_api(api_choice, prompt)


async def create_lore_material(detailed_world: str, api_choice: str) -> Dict[str, List[str]]:
    """Extract key elements from the detailed world to create lore and background material."""
    prompt = f"""
    Based on the detailed world description below, extract key elements to create lore and background material.
    Organize the lore into the following categories:
    1. Historical events
    2. Character profiles
    3. Location descriptions
    4. Cultural practices
    5. Technologies or magical systems

    Detailed World:
    {detailed_world}

    For each category, provide at least 5 concise but informative entries.
    Format your response as a JSON object with the categories as keys and lists of entries as values.
    """
    response = await call_api(api_choice, prompt)
    lore_material = parse_json_response(response)

    # Ensure the response has the expected structure
    expected_categories = ["Historical events", "Character profiles", "Location descriptions", "Cultural practices",
                           "Technologies or magical systems"]
    for category in expected_categories:
        if category not in lore_material or not isinstance(lore_material[category], list):
            lore_material[category] = [f"{category} entry {i + 1}" for i in range(5)]

    return lore_material


async def generate_documents(lore_material: Dict[str, List[str]], api_choice: str) -> Dict[str, List[str]]:
    """Generate primary, secondary, and tertiary documents based on the lore material."""
    documents = {}

    for doc_type, count in [("primary", 5), ("secondary", 7), ("tertiary", 10)]:
        prompt = f"""
        Using the following lore material, generate {count} {doc_type} documents that relate to the story.
        {doc_type.capitalize()} documents should be {"core materials directly related to the main plot" if doc_type == "primary" else "materials providing context or background" if doc_type == "secondary" else "peripheral items adding depth to the world"}.

        Lore Material:
        {json.dumps(lore_material, indent=2)}

        Format your response as a JSON array of strings, where each string is the content of a {doc_type} document.
        """
        response = await call_api(api_choice, prompt)
        parsed_response = parse_json_response(response)

        # Ensure we have the correct number of documents
        if isinstance(parsed_response, list):
            documents[doc_type] = parsed_response[:count]
        else:
            documents[doc_type] = [f"{doc_type.capitalize()} document {i + 1}" for i in range(count)]

        # Pad the list if we don't have enough documents
        while len(documents[doc_type]) < count:
            documents[doc_type].append(f"{doc_type.capitalize()} document {len(documents[doc_type]) + 1}")

    return documents


async def craft_letters_to_show(detailed_world: str, api_choice: str) -> List[str]:
    """Craft letters to the show to subtly guide the hosts' discussions."""
    prompt = f"""
    Based on the detailed world description below, write 5 letters to the show 'Deep Dive'.
    These letters should subtly guide the hosts' discussions and hint at episodic themes.
    In each letter:
    1. Reference 'Host 1' and 'Host 2' by name
    2. Mention a specific aspect of the story or world
    3. Pose a question or theory that could spark discussion
    4. Use the phrase 'Deep Dive' at least once

    Detailed World:
    {detailed_world}

    Format your response as a JSON array of strings, where each string is the content of a letter.
    """
    response = await call_api(api_choice, prompt)
    letters = parse_json_response(response)

    # Ensure we have 5 letters
    if not isinstance(letters, list) or len(letters) < 5:
        letters = [f"Letter to Deep Dive show {i + 1}" for i in range(5)]

    return letters[:5]


def organize_into_episodes(documents: Dict[str, List[str]], letters_to_show: List[str]) -> List[EpisodeType]:
    """Organize documents and letters into episodes, maintaining tension and surprises."""
    num_episodes = min(len(documents['primary']), len(letters_to_show))
    episodes = []

    for i in range(num_episodes):
        episode = {
            'primary': documents['primary'][i],
            'secondary': documents['secondary'][i * 2:(i + 1) * 2] if i * 2 < len(documents['secondary']) else [],
            'tertiary': documents['tertiary'][i * 3:(i + 1) * 3] if i * 3 < len(documents['tertiary']) else [],
            'letter': letters_to_show[i]
        }
        episodes.append(episode)

    return episodes


async def evaluate_and_refine(episodes: List[EpisodeType], api_choice: str) -> List[EpisodeType]:
    """Evaluate the coherence of episodes and refine if necessary."""
    refined_episodes = []

    for i, episode in enumerate(episodes):
        evaluation_prompt = f"""
        Evaluate the coherence and impact of the following episode materials:

        Primary Document: {episode['primary']}
        Secondary Documents: {episode['secondary']}
        Tertiary Documents: {episode['tertiary']}
        Letter to Show: {episode['letter']}

        Provide a JSON object with the following keys:
        - "is_coherent": boolean indicating if the episode is coherent
        - "suggestions": list of suggestions for improvement if not coherent
        """
        response = await call_api(api_choice, evaluation_prompt)
        evaluation = parse_json_response(response)

        if not isinstance(evaluation, dict) or 'is_coherent' not in evaluation:
            evaluation = {"is_coherent": False, "suggestions": ["Improve coherence"]}

        if not evaluation.get('is_coherent', False):
            refinement_prompt = f"""
            Refine the following episode materials based on these suggestions:
            {json.dumps(evaluation.get('suggestions', []), indent=2)}

            Current Episode:
            {json.dumps(episode, indent=2)}

            Provide a refined version of the episode, maintaining the same structure but improving coherence and impact.
            """
            response = await call_api(api_choice, refinement_prompt)
            refined_episode = parse_json_response(response)
            if isinstance(refined_episode, dict) and all(key in refined_episode for key in episode.keys()):
                refined_episodes.append(refined_episode)
            else:
                refined_episodes.append(episode)  # Fallback to original if refinement fails
        else:
            refined_episodes.append(episode)

    return refined_episodes


async def generate_story_content(story_details: str, api_choice: str, state: Optional[StateType] = None) -> StateType:
    """Generate all story content, with the ability to resume from a saved state."""
    if state is None:
        state = {}

    steps = [
        ("detailed_world", generate_detailed_world),
        ("lore_material", create_lore_material),
        ("documents", generate_documents),
        ("letters_to_show", craft_letters_to_show),
        ("episodes", organize_into_episodes),
        ("refined_episodes", evaluate_and_refine)
    ]

    for step_name, step_function in steps:
        if step_name not in state:
            logging.info(f"Generating {step_name}...")
            if step_name == "episodes":
                state[step_name] = step_function(state["documents"], state["letters_to_show"])
            elif step_name == "refined_episodes":
                state[step_name] = await step_function(state["episodes"], api_choice)
            elif step_name in ["detailed_world", "letters_to_show"]:
                state[step_name] = await step_function(story_details, api_choice)
            elif step_name == "lore_material":
                state[step_name] = await step_function(state["detailed_world"], api_choice)
            elif step_name == "documents":
                state[step_name] = await step_function(state["lore_material"], api_choice)
            save_state(state)
        else:
            logging.info(f"Skipping {step_name} generation (already exists in state)")

    return state


def save_output(state: StateType, output_dir: Path):
    """Save the generate