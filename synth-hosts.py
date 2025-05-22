# Patched by Document Generator Patcher v1.0.0 on 2024-12-16 21:19:00
# Patched by Document Generator Patcher v1.0.0 on 2024-12-16 21:18:43
# story_document_generator.py

import os
import json
import time
import logging
import textwrap
import csv
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import aiofiles
from openai import AsyncOpenAI, OpenAI
import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
import re


# Custom JSON encoder for serialization
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for our data classes and custom types."""
    def default(self, obj):
        if isinstance(obj, WorldBuildingDetail):
            return {
                "category": obj.category,
                "content": obj.content,
                "importance": obj.importance,
                "connected_elements": obj.connected_elements
            }
        elif isinstance(obj, DocumentType):
            return {
                "name": obj.name,
                "description": obj.description,
                "detail_level": obj.detail_level,
                "category": obj.category,
                "estimated_length": obj.estimated_length,
                "format": obj.format
            }
        elif isinstance(obj, Episode):
            return {
                "number": obj.number,
                "theme": obj.theme,
                "documents": obj.documents,
                "letters": obj.letters,
                "estimated_duration": obj.estimated_duration
            }
        elif isinstance(obj, StoryWorld):
            return {
                "description": obj.description,
                "history": obj.history,
                "characters": obj.characters,
                "locations": obj.locations,
                "events": obj.events,
                "timeline": obj.timeline
            }
        return super().default(obj)



# Type aliases and data classes
@dataclass
class DocumentType:
    name: str
    description: str
    detail_level: str
    category: str  # primary, secondary, or tertiary
    estimated_length: int  # approximate word count
    format: str  # e.g., "text", "dialogue", "transcript", etc.


@dataclass
class Episode:
    number: int
    theme: str
    documents: List[str]
    letters: List[str]
    estimated_duration: int  # in minutes


@dataclass
class WorldBuildingDetail:
    category: str
    content: str
    importance: int  # 1-10 scale
    connected_elements: List[str]


class StoryWorld:
    def __init__(self, base_description: str):
        self.description = base_description
        self.history: List[WorldBuildingDetail] = []
        self.characters: Dict[str, Dict[str, Any]] = {}
        self.locations: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.timeline: Dict[str, List[str]] = {}

    def add_character(self, name: str, details: Dict[str, Any]) -> None:
        self.characters[name] = details

    def add_location(self, name: str, details: Dict[str, Any]) -> None:
        self.locations[name] = details

    def add_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    def update_timeline(self, date: str, events: List[str]) -> None:
        if date in self.timeline:
            self.timeline[date].extend(events)
        else:
            self.timeline[date] = events


# Constants
class Constants:
    OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
    NEBIUS_API_KEY_ENV_VAR = "NEBIUS_API_KEY"
    ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
    OUTPUT_DIR = Path("generated_documents")
    RUNS_RECORD_FILE = Path("runs_record.csv")
    LOGS_DIR = Path("logs")
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # Seconds
    DEFAULT_MAX_TOKENS = 15000
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_PARALLEL_CALLS = True

    # Model configurations
    MODEL_CONFIGS = {
        "openai": {
            "name": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 0.7
        },
        "nebius": {
            "name": "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "max_tokens": 2048,
            "temperature": 0.7
        },
        "anthropic": {
            "name": "claude-3-sonnet-20240229",
            "max_tokens": 4000,
            "temperature": 0.7
        },
        "ollama": {
            "name": "llama3.2",
            "max_tokens": 2048,
            "temperature": 0.7
        }
    }


# Custom Exceptions
class DocumentGeneratorError(Exception):
    """Base exception for document generator errors."""
    pass


class APIKeyError(DocumentGeneratorError):
    """Raised when API keys are missing or invalid."""
    pass


class APICallError(DocumentGeneratorError):
    """Raised when API calls fail."""
    pass


class DocumentGenerationError(DocumentGeneratorError):
    """Raised when document generation fails."""
    pass


# Logger setup
def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    Constants.LOGS_DIR.mkdir(exist_ok=True)
    log_file = Constants.LOGS_DIR / f"generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("document_generator")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# API Clients
class APIClient:
    """Base class for API clients."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger("document_generator")

    async def call_api(self, prompt: str, **kwargs) -> str:
        """Template method for API calls."""
        raise NotImplementedError


class OpenAIClient(APIClient):
    """OpenAI API client implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = AsyncOpenAI(api_key=api_key)

    async def call_api(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get('max_tokens', Constants.MODEL_CONFIGS['openai']['max_tokens'])
        temperature = kwargs.get('temperature', Constants.MODEL_CONFIGS['openai']['temperature'])

        for attempt in range(Constants.MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=Constants.MODEL_CONFIGS['openai']['name'],
                    messages=[
                        {"role": "system",
                         "content": "You are a creative writing assistant specializing in generating detailed story documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.error(f"OpenAI API error on attempt {attempt + 1}: {str(e)}")
                if attempt < Constants.MAX_RETRIES - 1:
                    await asyncio.sleep(Constants.RETRY_BACKOFF * (attempt + 1))
                else:
                    raise APICallError(f"OpenAI API call failed after {Constants.MAX_RETRIES} attempts") from e


class AnthropicClient(APIClient):
    """Anthropic API client implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    async def call_api(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get('max_tokens', Constants.MODEL_CONFIGS['anthropic']['max_tokens'])
        temperature = kwargs.get('temperature', Constants.MODEL_CONFIGS['anthropic']['temperature'])

        for attempt in range(Constants.MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=Constants.MODEL_CONFIGS['anthropic']['name'],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            except Exception as e:
                self.logger.error(f"Anthropic API error on attempt {attempt + 1}: {str(e)}")
                if attempt < Constants.MAX_RETRIES - 1:
                    await asyncio.sleep(Constants.RETRY_BACKOFF * (attempt + 1))
                else:
                    raise APICallError(f"Anthropic API call failed after {Constants.MAX_RETRIES} attempts") from e


# Document Generator
class DocumentGenerator:
    """Main class for generating story documents."""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.logger = logging.getLogger("document_generator")
        self.console = Console()
        self.story_world: Optional[StoryWorld] = None

    # Update the generate_world method in the DocumentGenerator class

    async def generate_world(self, story_details: str) -> StoryWorld:
        """Generate detailed world building based on story description."""
        prompt = textwrap.dedent(f"""
        You are a creative writing assistant. Create a detailed world based on this story description.
        Provide your response in strict JSON format with the following structure:
        {{
            "history": [
                {{
                    "category": "string",
                    "content": "string",
                    "importance": integer between 1-10,
                    "connected_elements": ["string"]
                }}
            ],
            "characters": {{
                "character_name": {{
                    "background": "string",
                    "motivations": ["string"],
                    "relationships": {{"other_character": "relationship_description"}},
                    "arc": "string"
                }}
            }},
            "locations": {{
                "location_name": {{
                    "description": "string",
                    "significance": "string",
                    "connected_characters": ["string"]
                }}
            }},
            "events": [
                {{
                    "name": "string",
                    "description": "string",
                    "impact": "string",
                    "connected_characters": ["string"],
                    "location": "string",
                    "date": "string"
                }}
            ]
        }}

        Story Description:
        {story_details}

        Remember:
        1. Use only the exact structure shown above
        2. Ensure all JSON keys are exactly as shown
        3. Use valid JSON format (double quotes, no trailing commas)
        4. Make the content rich and detailed but maintain valid JSON syntax
        """)

        try:
            # Get raw response from API
            response = await self.api_client.call_api(prompt)

            # Log the raw response for debugging
            self.logger.debug(f"Raw API response:\n{response}")

            # Try to extract JSON from the response if it's wrapped in markdown or other text
            try:
                # Find the first { and last } to extract potential JSON
                start_idx = response.find('{')
                end_idx = response.rfind('}')

                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                else:
                    raise ValueError("No JSON structure found in response")

                # Try to parse the extracted JSON
                world_data = json.loads(json_str)

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse response as JSON: {str(e)}")
                self.logger.debug("Attempting fallback parsing...")

                # Fallback: Try to construct a valid JSON structure
                fallback_world = {
                    "history": [
                        {
                            "category": "Background",
                            "content": story_details,
                            "importance": 10,
                            "connected_elements": []
                        }
                    ],
                    "characters": {
                        "Placeholder": {
                            "background": "Generated from story description",
                            "motivations": ["To be determined"],
                            "relationships": {},
                            "arc": "To be developed"
                        }
                    },
                    "locations": {
                        "Main Setting": {
                            "description": "Primary story location",
                            "significance": "Central to the narrative",
                            "connected_characters": []
                        }
                    },
                    "events": [
                        {
                            "name": "Story Beginning",
                            "description": story_details,
                            "impact": "Initiates the narrative",
                            "connected_characters": [],
                            "location": "Main Setting",
                            "date": "Start"
                        }
                    ]
                }

                self.logger.warning("Using fallback world structure due to parsing error")
                world_data = fallback_world

            # Create and populate StoryWorld object
            story_world = StoryWorld(story_details)

            # Safely populate history
            if "history" in world_data:
                for history_item in world_data["history"]:
                    try:
                        # Ensure all required fields are present with default values if needed
                        processed_item = {
                            "category": history_item.get("category", "Unknown"),
                            "content": history_item.get("content", ""),
                            "importance": int(history_item.get("importance", 5)),
                            "connected_elements": history_item.get("connected_elements", [])
                        }
                        story_world.history.append(WorldBuildingDetail(**processed_item))
                    except Exception as e:
                        self.logger.warning(f"Failed to process history item: {str(e)}")
                        continue

            # Safely populate other attributes
            story_world.characters = world_data.get("characters", {})
            story_world.locations = world_data.get("locations", {})
            story_world.events = world_data.get("events", [])

            # Build timeline from events
            for event in story_world.events:
                if isinstance(event, dict) and "date" in event and "name" in event:
                    story_world.update_timeline(event["date"], [event["name"]])

            self.story_world = story_world
            return story_world

        except Exception as e:
            self.logger.error(f"World generation failed: {str(e)}", exc_info=True)
            raise DocumentGenerationError(f"Failed to generate world: {str(e)}")

            async def generate_document_types(self, story_world: StoryWorld) -> List[DocumentType]:
            """Generate document types based on the story world."""
        self.logger.info("Starting document type generation")
        
        prompt = textwrap.dedent(f"""
        You are a document generation assistant. Based on this story world, create different types of documents 
        that can tell the story indirectly. Include primary (direct evidence), secondary (indirect evidence), 
        and tertiary (background context) documents.

        Story World Summary:
        - Description: {story_world.description}
        - Characters: {len(story_world.characters)}
        - Locations: {len(story_world.locations)}
        - Events: {len(story_world.events)}

        For each document type, provide:
        1. name: The type of document
        2. description: Its purpose in the story
        3. detail_level: Expected length (brief, moderate, detailed)
        4. category: primary, secondary, or tertiary
        5. estimated_length: Approximate word count
        6. format: The document format (letter, report, transcript, etc.)

        Provide at least 6 document types, with a mix of categories.
        Format as JSON array.
        """)

        try:
            # Get response from API
            response = await self.api_client.call_api(prompt)
            self.logger.debug(f"Raw document types response:
{response}")

            try:
                # Try to find and extract JSON array
                json_match = re.search(r'(\[.*?\])', response, re.DOTALL | re.MULTILINE)
                if json_match:
                    json_str = json_match.group(1)
                    doc_types_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON array found in response")

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse JSON response: {str(e)}")
                
                # Provide fallback document types with story-specific names
                doc_types_data = [
                    {
                        "name": f"{next(iter(story_world.characters.keys()), 'Main Character')}'s Journal",
                        "description": "Personal diary entries from main character",
                        "detail_level": "detailed",
                        "category": "primary",
                        "estimated_length": 800,
                        "format": "journal"
                    },
                    {
                        "name": f"Report on {next(iter(story_world.events), {}).get('name', 'Key Event')}",
                        "description": "Official documentation of key story event",
                        "detail_level": "moderate",
                        "category": "secondary",
                        "estimated_length": 500,
                        "format": "report"
                    },
                    {
                        "name": f"History of {next(iter(story_world.locations.keys()), 'Main Location')}",
                        "description": "Background information about key location",
                        "detail_level": "detailed",
                        "category": "tertiary",
                        "estimated_length": 1000,
                        "format": "document"
                    }
                ]

            validated_doc_types = []
            categories_count = {"primary": 0, "secondary": 0, "tertiary": 0}
            
            for doc in doc_types_data:
                try:
                    validated_doc = {
                        "name": str(doc.get("name", "Unknown Document")),
                        "description": str(doc.get("description", "Additional story content")),
                        "detail_level": str(doc.get("detail_level", "moderate")),
                        "category": str(doc.get("category", "secondary")),
                        "estimated_length": int(doc.get("estimated_length", 500)),
                        "format": str(doc.get("format", "text"))
                    }
                    
                    # Track category counts
                    categories_count[validated_doc["category"]] += 1
                    
                    validated_doc_types.append(DocumentType(**validated_doc))
                except Exception as e:
                    self.logger.warning(f"Failed to validate document type: {str(e)}")
                    continue

            if not validated_doc_types:
                raise ValueError("No valid document types were generated")
                
            # Log success with category distribution
            self.logger.info(f"Generated {len(validated_doc_types)} document types: {categories_count}")
            
            return validated_doc_types

        except Exception as e:
            self.logger.error(f"Failed to generate document types: {str(e)}")
            raise DocumentGenerationError(f"Failed to generate document types: {str(e)}")
async def generate_episodes(self, story_world: StoryWorld, doc_types: List[DocumentType],
                              num_episodes: int) -> List[Episode]:
        """Generate episode structure and content."""
        prompt = textwrap.dedent(f"""
        Create {num_episodes} episodes that gradually reveal the story through the available documents.
        Each episode should have a central theme and include a selection of documents that build tension
        and maintain audience engagement.

        Story World:
        {json.dumps(story_world.__dict__, indent=2, cls=CustomJSONEncoder)}

        Available Document Types:
        {json.dumps([doc_type.__dict__ for doc_type in doc_types], indent=2, cls=CustomJSONEncoder)}

        Format the response as a JSON array of objects with the following structure:
        [{{
            "number": int,
            "theme": str,
            "documents": [str],
            "letters": [str],
            "estimated_duration": int
        }}]
        """)

        try:
            response = await self.api_client.call_api(prompt)
            episodes_data = json.loads(response)
            return [Episode(**episode) for episode in episodes_data]
        except Exception as e:
            raise DocumentGenerationError(f"Failed to generate episodes: {str(e)}")

    async def generate_document_content(self, doc_type: DocumentType, story_world: StoryWorld,
                                     episode: Episode) -> str:
        """Generate content for a specific document."""
        prompt = textwrap.dedent(f"""
        Generate content for a {doc_type.name} document with the following specifications:

        Document Type: {doc_type.description}
        Detail Level: {doc_type.detail_level}
        Format: {doc_type.format}
        Estimated Length: {doc_type.estimated_length} words

        Episode Theme: {episode.theme}

        Story World Context:
        {json.dumps(story_world.__dict__, indent=2, cls=CustomJSONEncoder)}

        The document should:
        1. Match the specified format and detail level
        2. Contribute to the episode's theme
        3. Reveal information naturally and indirectly
        4. Maintain authenticity for its document type
        5. Connect to other story elements when appropriate

        Generate the document content:
        """)

        try:
            return await self.api_client.call_api(
                prompt,
                max_tokens=doc_type.estimated_length * 4,  # Rough estimate of tokens needed
                temperature=0.8  # Slightly higher temperature for more creative variation
            )
        except Exception as e:
            raise DocumentGenerationError(f"Failed to generate content for {doc_type.name}: {str(e)}")


async def main():
    """Main execution function."""
    # Setup
    logger = setup_logging()
    console = Console()

    try:
        # Load API keys
        openai_key = os.getenv(Constants.OPENAI_API_KEY_ENV_VAR)
        anthropic_key = os.getenv(Constants.ANTHROPIC_API_KEY_ENV_VAR)

        if not openai_key or not anthropic_key:
            raise APIKeyError("Missing required API keys")

        # Initialize API client (using OpenAI for this example)
        api_client = OpenAIClient(openai_key)

        # Initialize document generator
        generator = DocumentGenerator(api_client)

        # Create rich interface
        console.print("[bold blue]Story Document Generator[/bold blue]", justify="center")
        console.print("=" * 50, justify="center")

        # Get story input
        story_details = Prompt.ask("\nEnter a brief description of your story")
        num_episodes = int(Prompt.ask("\nHow many episodes would you like to generate", default="5"))

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
        ) as progress:
            # Generate story world
            task = progress.add_task("[cyan]Generating story world...", total=None)
            story_world = await generator.generate_world(story_details)
            progress.update(task, completed=True)

            # Display world summary
            console.print("\n[bold green]Story World Generated:[/bold green]")
            world_table = Table(show_header=True, header_style="bold magenta")
            world_table.add_column("Category", style="dim")
            world_table.add_column("Count")
            world_table.add_row("Characters", str(len(story_world.characters)))
            world_table.add_row("Locations", str(len(story_world.locations)))
            world_table.add_row("Events", str(len(story_world.events)))
            console.print(world_table)

            # Generate document types
            task = progress.add_task("[cyan]Generating document types...", total=1)
            try:
                doc_types = await generator.generate_document_types(story_world)
                if not doc_types:
                    raise DocumentGenerationError("No document types were generated")
                progress.update(task, advance=1)
            except Exception as e:
                progress.update(task, completed=True)
                logger.error(f"Failed to generate document types: {str(e)}")
                console.print("[red]Failed to generate document types. Check logs for details.[/red]")
                raise
            progress.update(task, completed=True)
            
            # Display document types summary
            console.print("
[bold green]Document Types Generated:[ /bold green]")

            # Display document types summary
            console.print("\n[bold green]Document Types Generated:[/bold green]")
            doc_table = Table(show_header=True, header_style="bold magenta")
            doc_table.add_column("Category")
            doc_table.add_column("Count")
            doc_counts = {"primary": 0, "secondary": 0, "tertiary": 0}
            for doc_type in doc_types:
                doc_counts[doc_type.category] += 1
            for category, count in doc_counts.items():
                doc_table.add_row(category.capitalize(), str(count))
            console.print(doc_table)

            # Generate episodes
            task = progress.add_task("[cyan]Generating episode structure...", total=None)
            episodes = await generator.generate_episodes(story_world, doc_types, num_episodes)
            progress.update(task, completed=True)

            # Create output directory
            run_id = str(uuid.uuid4())[:8]
            output_dir = Constants.OUTPUT_DIR / f"run_{run_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate and save documents for each episode
            for episode in episodes:
                episode_dir = output_dir / f"episode_{episode.number}"
                episode_dir.mkdir(exist_ok=True)

                console.print(f"\n[bold blue]Generating Episode {episode.number}[/bold blue]")
                console.print(f"Theme: {episode.theme}")

                task = progress.add_task(
                    f"[cyan]Generating documents for Episode {episode.number}...",
                    total=len(episode.documents)
                )

                for doc_name in episode.documents:
                    # Find corresponding document type
                    doc_type = next((dt for dt in doc_types if dt.name == doc_name), None)
                    if doc_type:
                        try:
                            # Generate document content
                            content = await generator.generate_document_content(
                                doc_type, story_world, episode
                            )

                            # Save document
                            doc_path = episode_dir / f"{doc_name.lower().replace(' ', '_')}.txt"
                            async with aiofiles.open(doc_path, 'w') as f:
                                await f.write(content)

                            progress.advance(task)

                        except DocumentGenerationError as e:
                            logger.error(f"Failed to generate document {doc_name}: {str(e)}")
                            console.print(f"[red]Error generating {doc_name}[/red]")

                # Generate and save episode letters
                letters_dir = episode_dir / "letters"
                letters_dir.mkdir(exist_ok=True)

                for i, letter_content in enumerate(episode.letters, 1):
                    letter_path = letters_dir / f"letter_{i}.txt"
                    async with aiofiles.open(letter_path, 'w') as f:
                        await f.write(letter_content)

            # Save story world data
            world_path = output_dir / "story_world.json"
            async with aiofiles.open(world_path, 'w') as f:
                await f.write(json.dumps(story_world.__dict__, indent=2, cls=CustomJSONEncoder))

            # Generate README
            readme_content = textwrap.dedent(f"""
                # Story Document Generation - Run {run_id}

                Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                ## Story Description
                {story_details}

                ## Episodes Generated: {num_episodes}

                ## Document Types
                - Primary: {doc_counts['primary']}
                - Secondary: {doc_counts['secondary']}
                - Tertiary: {doc_counts['tertiary']}

                ## Directory Structure
                - episode_X/: Episode-specific documents
                  - letters/: Episode-specific letters
                - story_world.json: Complete story world data

                Generated using Story Document Generator
                """)

            readme_path = output_dir / "README.md"
            async with aiofiles.open(readme_path, 'w') as f:
                await f.write(readme_content)

            console.print("\n[bold green]Generation Complete![/bold green]")
            console.print(f"Output directory: {output_dir}")

    except APIKeyError as e:
        logger.error(f"API Key Error: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/red]")
    except DocumentGeneratorError as e:
        logger.error(f"Document Generation Error: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/red]")
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        console.print("[red]An unexpected error occurred. Check the logs for details.[/red]")


if __name__ == "__main__":
    asyncio.run(main())

