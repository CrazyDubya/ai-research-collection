import sys
from pathlib import Path
import datetime
import re

def get_fixed_progress_section():
    return '''            # Generate document types
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
            console.print("\\n[bold green]Document Types Generated:[/bold green]")
'''

def get_fixed_method():
    return '''    async def generate_document_types(self, story_world: StoryWorld) -> List[DocumentType]:
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
            self.logger.debug(f"Raw document types response:\\n{response}")

            try:
                # Try to find and extract JSON array
                json_match = re.search(r'(\\[.*?\\])', response, re.DOTALL | re.MULTILINE)
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
'''

def patch_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create backup
    backup_path = f"{file_path}.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Fix the progress bar section
    progress_section_pattern = r'# Generate document types.*?progress\.update\(task, completed=True\)'
    content = re.sub(
        progress_section_pattern,
        get_fixed_progress_section().strip(),
        content,
        flags=re.DOTALL
    )

    # Fix the method implementation
    method_pattern = r'async def generate_document_types.*?(?=async def|$)'
    content = re.sub(
        method_pattern,
        get_fixed_method(),
        content,
        flags=re.DOTALL
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return backup_path

def main():
    if len(sys.argv) != 2:
        print("Usage: python progress_fix.py <path_to_script.py>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    try:
        backup_path = patch_file(file_path)
        print("\n✅ Progress Bar Integration Fix")
        print("=" * 30)
        print("\nChanges made:")
        print("  1. Fixed progress bar implementation")
        print("  2. Added better error handling")
        print("  3. Improved logging")
        print("  4. Enhanced document type generation")
        print("  5. Added category tracking")
        print("\nBackup created:", backup_path)
        print("\nNext steps:")
        print("1. Review the changes")
        print("2. Test the document generation")
        print("3. Check the progress bar behavior")
        print("4. Verify error handling")
    except Exception as e:
        print(f"Error: Failed to patch file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()