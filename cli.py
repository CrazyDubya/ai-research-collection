#!/usr/bin/env python3
"""
Unified Command Line Interface for AI Research Collection
Provides interactive access to boolean logic experiments, agent systems, and story generation.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for enhanced CLI experience: pip install rich")

from config_manager import ConfigManager, get_config
from boolean_logic_lib import BooleanGateLibrary, TokenClassifier, BooleanLogicExperiment, generate_truth_tables

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIResearchCLI:
    """Main CLI application class."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.config = get_config()
        self.config.load_config()
        
    def print(self, *args, **kwargs):
        """Enhanced print with rich support."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)
    
    def prompt(self, message: str, choices: Optional[List[str]] = None) -> str:
        """Enhanced prompt with rich support."""
        if RICH_AVAILABLE and choices:
            return Prompt.ask(message, choices=choices)
        elif RICH_AVAILABLE:
            return Prompt.ask(message)
        else:
            if choices:
                print(f"{message} (choices: {', '.join(choices)})")
            response = input(f"{message}: ")
            if choices and response not in choices:
                print(f"Invalid choice. Please select from: {', '.join(choices)}")
                return self.prompt(message, choices)
            return response
    
    def confirm(self, message: str) -> bool:
        """Enhanced confirmation with rich support."""
        if RICH_AVAILABLE:
            return Confirm.ask(message)
        else:
            response = input(f"{message} (y/n): ").lower().strip()
            return response in ('y', 'yes', '1', 'true')
    
    def show_main_menu(self):
        """Display the main menu."""
        if self.console:
            table = Table(title="AI Research Collection - Main Menu")
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description", style="magenta")
            
            table.add_row("1", "Boolean Logic Experiments")
            table.add_row("2", "Agent System Management")
            table.add_row("3", "Story Generation")
            table.add_row("4", "Configuration Management")
            table.add_row("5", "Documentation & Help")
            table.add_row("6", "Exit")
            
            self.console.print(table)
        else:
            print("\n=== AI Research Collection - Main Menu ===")
            print("1. Boolean Logic Experiments")
            print("2. Agent System Management")
            print("3. Story Generation")
            print("4. Configuration Management")
            print("5. Documentation & Help")
            print("6. Exit")
    
    def run_boolean_experiments(self):
        """Run boolean logic experiments."""
        self.print("\n[bold cyan]Boolean Logic Experiments[/bold cyan]" if RICH_AVAILABLE else "\n=== Boolean Logic Experiments ===")
        
        # Initialize components
        gate_lib = BooleanGateLibrary()
        classifier = TokenClassifier(
            logical_words=self.config.get('boolean_logic.logical_words'),
            emotional_words=self.config.get('boolean_logic.emotional_words'),
            logical_phrases=self.config.get('boolean_logic.logical_phrases'),
            emotional_phrases=self.config.get('boolean_logic.emotional_phrases')
        )
        
        while True:
            choice = self.prompt(
                "Choose action",
                ["truth_tables", "classify_text", "run_experiment", "back"]
            )
            
            if choice == "truth_tables":
                self.print("\n[bold]Boolean Gate Truth Tables[/bold]" if RICH_AVAILABLE else "\n=== Boolean Gate Truth Tables ===")
                tables = generate_truth_tables()
                self.print(tables)
            
            elif choice == "classify_text":
                text = self.prompt("Enter text to classify")
                if text:
                    # Simple tokenization for demo
                    tokens = text.split()
                    classifications = classifier.classify_tokens_batch(tokens, text)
                    
                    if RICH_AVAILABLE:
                        table = Table(title="Token Classification Results")
                        table.add_column("Token", style="cyan")
                        table.add_column("Classification", style="magenta")
                        
                        for token, classification in zip(tokens, classifications):
                            color = "green" if classification == "logical" else "red" if classification == "emotional" else "white"
                            table.add_row(token, f"[{color}]{classification}[/{color}]")
                        
                        self.console.print(table)
                    else:
                        print("\nClassification Results:")
                        for token, classification in zip(tokens, classifications):
                            print(f"{token}: {classification}")
            
            elif choice == "run_experiment":
                self.print("[yellow]Note: Full experiments require model loading (may take time)[/yellow]" if RICH_AVAILABLE else "Note: Full experiments require model loading")
                gate = self.prompt("Choose gate", gate_lib.list_gates())
                text = self.prompt("Enter test text")
                logical_input = self.confirm("Is logical input True?")
                emotional_input = self.confirm("Is emotional input True?")
                
                # Create experiment (without models for quick demo)
                experiment = BooleanLogicExperiment(gate_lib, classifier)
                result = experiment.run_gate_experiment(gate, text, logical_input, emotional_input)
                
                self.print(f"\nGate {gate} result: {result['gate_result']}")
                self.print(f"Text: {text}")
                self.print(f"Inputs: logical={logical_input}, emotional={emotional_input}")
            
            elif choice == "back":
                break
    
    def manage_agents(self):
        """Manage agent systems."""
        self.print("\n[bold cyan]Agent System Management[/bold cyan]" if RICH_AVAILABLE else "\n=== Agent System Management ===")
        self.print("[yellow]Agent functionality requires TinyTroupe installation[/yellow]" if RICH_AVAILABLE else "Note: Agent functionality requires TinyTroupe installation")
        
        faculties = self.config.get('agents.default_faculties', [])
        self.print(f"Default agent faculties: {', '.join(faculties)}")
        
        # Check if LitAg.py is available
        if Path("LitAg.py").exists():
            if self.confirm("Run LitAg.py demo?"):
                self.print("Running LitAg.py...")
                try:
                    # Note: This would need proper import handling for TinyTroupe
                    self.print("[red]TinyTroupe import required - please install manually[/red]" if RICH_AVAILABLE else "TinyTroupe import required")
                except ImportError as e:
                    self.print(f"[red]Import error: {e}[/red]" if RICH_AVAILABLE else f"Import error: {e}")
        
        self.prompt("Press Enter to continue", [])
    
    def generate_stories(self):
        """Story generation interface."""
        self.print("\n[bold cyan]Story Generation[/bold cyan]" if RICH_AVAILABLE else "\n=== Story Generation ===")
        
        # List available story scripts
        story_scripts = [f for f in os.listdir('.') if f.startswith('lisa_diary') and f.endswith('.py')]
        
        if story_scripts:
            self.print(f"Available story scripts: {', '.join(story_scripts)}")
            
            if self.confirm("Would you like to run a story generation script?"):
                script = self.prompt("Choose script", story_scripts + ["back"])
                if script != "back" and script in story_scripts:
                    self.print(f"Running {script}...")
                    self.print("[yellow]Note: Story scripts may require additional setup[/yellow]" if RICH_AVAILABLE else "Note: Story scripts may require additional setup")
        else:
            self.print("No story generation scripts found")
        
        self.prompt("Press Enter to continue", [])
    
    def manage_config(self):
        """Configuration management interface."""
        self.print("\n[bold cyan]Configuration Management[/bold cyan]" if RICH_AVAILABLE else "\n=== Configuration Management ===")
        
        while True:
            choice = self.prompt(
                "Choose action",
                ["view_config", "edit_config", "save_config", "create_sample", "back"]
            )
            
            if choice == "view_config":
                if RICH_AVAILABLE:
                    panel = Panel(str(self.config.config), title="Current Configuration")
                    self.console.print(panel)
                else:
                    print("Current Configuration:")
                    print(self.config.config)
            
            elif choice == "edit_config":
                self.print("Interactive configuration editing:")
                section = self.prompt("Choose section", ["boolean_logic", "models", "experiments", "agents", "cli"])
                
                if section == "boolean_logic":
                    new_word = self.prompt("Add logical word (or press Enter to skip)")
                    if new_word:
                        self.config.add_to_list("boolean_logic.logical_words", new_word)
                        self.print(f"Added '{new_word}' to logical words")
            
            elif choice == "save_config":
                if self.config.save_config():
                    self.print("[green]Configuration saved successfully[/green]" if RICH_AVAILABLE else "Configuration saved successfully")
                else:
                    self.print("[red]Error saving configuration[/red]" if RICH_AVAILABLE else "Error saving configuration")
            
            elif choice == "create_sample":
                self.config.create_sample_config()
                self.print("[green]Sample configuration created in config/sample_config.yaml[/green]" if RICH_AVAILABLE else "Sample configuration created")
            
            elif choice == "back":
                break
    
    def show_help(self):
        """Show documentation and help."""
        self.print("\n[bold cyan]Documentation & Help[/bold cyan]" if RICH_AVAILABLE else "\n=== Documentation & Help ===")
        
        help_text = """
        [bold]AI Research Collection Help[/bold]
        
        [cyan]Boolean Logic Experiments:[/cyan]
        - Test logical gates (AND, OR, XOR, etc.) with text classification
        - Classify tokens as logical, emotional, or neutral
        - Run multi-model experiments (requires model setup)
        
        [cyan]Agent Systems:[/cyan]
        - Manage AI agents with standard faculties (RECALL, CONSULT, THINK, TALK, DONE)
        - Requires TinyTroupe library installation
        
        [cyan]Story Generation:[/cyan]
        - Run interactive story generation scripts
        - Character-based narrative systems
        
        [cyan]Configuration:[/cyan]
        - Customize classification dictionaries
        - Manage model preferences
        - Export/import settings
        
        [cyan]Files Overview:[/cyan]
        - boolean_logic_lib.py: Core boolean logic functionality
        - config_manager.py: Configuration management
        - cli.py: This unified interface
        """
        
        if RICH_AVAILABLE:
            self.console.print(help_text)
        else:
            print(help_text.replace('[bold]', '').replace('[/bold]', '').replace('[cyan]', '').replace('[/cyan]', ''))
        
        self.prompt("Press Enter to continue", [])
    
    def run(self):
        """Main application loop."""
        self.print("\n[bold green]Welcome to AI Research Collection[/bold green]" if RICH_AVAILABLE else "Welcome to AI Research Collection")
        self.print("Unified interface for boolean logic, agents, and story generation\n")
        
        while True:
            self.show_main_menu()
            choice = self.prompt("Choose option", ["1", "2", "3", "4", "5", "6"])
            
            if choice == "1":
                self.run_boolean_experiments()
            elif choice == "2":
                self.manage_agents()
            elif choice == "3":
                self.generate_stories()
            elif choice == "4":
                self.manage_config()
            elif choice == "5":
                self.show_help()
            elif choice == "6":
                self.print("Goodbye!")
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Research Collection CLI")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("--demo", choices=["boolean", "agents", "help"], help="Run a specific demo")
    
    args = parser.parse_args()
    
    cli = AIResearchCLI()
    
    if args.config:
        cli.config.load_config(args.config)
    
    if args.demo:
        if args.demo == "boolean":
            cli.run_boolean_experiments()
        elif args.demo == "agents":
            cli.manage_agents()
        elif args.demo == "help":
            cli.show_help()
    elif args.non_interactive:
        print("Non-interactive mode - use --demo flag for specific functionality")
    else:
        cli.run()


if __name__ == "__main__":
    main()