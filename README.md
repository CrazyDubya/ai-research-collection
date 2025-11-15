# AI Research & Development Collection

A unified collection of AI research projects, experiments, and utilities focusing on boolean logic analysis, agent simulation, and document generation with a clean, extensible architecture.

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd ai-research-collection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run unified CLI
python cli.py
```

### Quick Demo
```bash
# Run boolean logic demo
python cli.py --demo boolean

# Show help and documentation
python cli.py --demo help

# Run tests
python test_ai_research.py
```

## 🏗️ Architecture Overview

### Core Components

#### **Boolean Logic Library** (`boolean_logic_lib.py`)
- Centralized boolean gate operations with consistent API
- Support for all standard gates (AND, OR, XOR, NOT, NAND, NOR, XNOR)
- Extensible token classification system
- Multi-model experiment framework
- Truth table generation

#### **Agent System** (`agent_system.py`)
- Standardized agent interface with unified faculties
- Standard faculties: RECALL, CONSULT, THINK, TALK, DONE
- Memory management with semantic indexing
- Agent orchestration for multi-agent conversations
- Specialized agents: LiteratureAgent, BooleanLogicAgent

#### **Configuration Management** (`config_manager.py`)
- YAML/JSON configuration support
- Runtime configuration updates
- Extensible classification dictionaries
- Model and experiment settings

#### **Unified CLI** (`cli.py`)
- Interactive terminal interface
- Boolean logic experiments
- Agent system management
- Configuration editing
- Rich terminal UI (optional)

## 📚 Main Features

### Boolean Logic Experiments
- **Unified Gate Library**: All boolean gates in one consistent API
- **Token Classification**: Classify text as logical, emotional, or neutral
- **Multi-Model Support**: Run experiments across multiple language models
- **Truth Table Generation**: Automatic markdown table generation
- **Configurable Dictionaries**: Runtime-extensible classification words/phrases

### Agent Systems
- **Standardized Interface**: Consistent API across all agent types
- **Memory Management**: Semantic memory with keyword-based recall
- **Faculty System**: Standard actions (RECALL, CONSULT, THINK, TALK, DONE)
- **Multi-Agent Orchestration**: Coordinate conversations between agents
- **Extensible Design**: Easy to add new agent types

### Configuration & Extensibility
- **YAML Configuration**: Human-readable configuration files
- **Runtime Updates**: Modify settings without restart
- **Plugin Architecture**: Add custom gates, classification types, agent faculties
- **Sample Configurations**: Generated examples for customization

## 🔧 Usage Examples

### Boolean Logic
```python
from boolean_logic_lib import BooleanGateLibrary, TokenClassifier

# Create gate library
gates = BooleanGateLibrary()
result = gates.evaluate_gate("XOR", True, False)  # Returns: True

# Classify tokens
classifier = TokenClassifier()
classification = classifier.classify_token("therefore")  # Returns: "logical"
```

### Agent System
```python
from agent_system import create_literature_agent, AgentOrchestrator

# Create and use an agent
agent = create_literature_agent("MyAgent")
agent.add_document("test.txt", "Sample document content")

action = agent.process_input("analyze the themes")
print(action.content)
```

### CLI Interface
```bash
# Interactive mode
python cli.py

# Boolean logic demo
python cli.py --demo boolean

# Custom configuration
python cli.py --config my_config.yaml
```

## 📁 Project Structure

### New Unified Components
- **boolean_logic_lib.py** - Consolidated boolean logic functionality
- **agent_system.py** - Unified agent interface and management
- **config_manager.py** - Configuration management system
- **cli.py** - Unified command-line interface
- **test_ai_research.py** - Comprehensive test suite
- **requirements.txt** - Dependency management

### Legacy Components (Available)
- **boolean_gates.py** - Original boolean gate implementation
- **boolean_logic_multimodel.py** - Original multi-model experiments
- **LitAg.py** - Original literature agent (TinyTroupe-based)
- **lisa_diary*.py** - Story generation experiments
- **chars/** - Character generation modules
- **void/** - Simulation engine experiments

### Configuration
- **config/** - Configuration files and samples
- **config/ai_research_config.yaml** - Main configuration
- **config/sample_config.yaml** - User customization template

## 🧪 Testing

```bash
# Run all tests
python test_ai_research.py

# Run specific test class
python -m unittest test_ai_research.TestBooleanGateLibrary

# Check functionality without ML dependencies
python test_ai_research.py  # Works without torch/transformers
```

## 🔄 Migration Guide

### From Original Boolean Logic Files
The new `boolean_logic_lib.py` provides the same functionality as the original files but with:
- Cleaner API
- Better error handling
- Extensible design
- Optional ML dependencies

### From Original Agent System
The new `agent_system.py` standardizes the interface while maintaining compatibility:
- Unified faculty system
- Better memory management
- Multi-agent coordination

## 🚀 Extensibility

### Adding Custom Gates
```python
from boolean_logic_lib import BooleanGateLibrary

gates = BooleanGateLibrary()
gates.add_custom_gate("IMPLIES", 2, lambda a, b: not a or b)
```

### Adding Custom Agent Types
```python
from agent_system import BaseAgent, AgentFaculty

class CustomAgent(BaseAgent):
    def _initialize_memory(self):
        # Add domain-specific knowledge
        pass
    
    def process_input(self, input_text):
        # Custom logic
        return self.execute_faculty(AgentFaculty.THINK, input_text)
```

### Extending Configuration
```python
from config_manager import get_config

config = get_config()
config.add_to_list("boolean_logic.logical_words", "algorithm")
config.save_config()
```

## 📖 Documentation

- **CLI Help**: `python cli.py --demo help`
- **API Documentation**: See docstrings in individual modules
- **Configuration**: Check `config/sample_config.yaml` for examples
- **Tests**: Review `test_ai_research.py` for usage patterns

## 🤝 Contributing

1. Follow the unified architecture patterns
2. Add tests for new functionality
3. Update configuration samples if needed
4. Maintain backward compatibility where possible
5. Document new features in README

## 📋 TODO / Roadmap

- [ ] GUI interface (Tkinter/PyQt/Web-based)
- [ ] CI/CD pipeline setup
- [ ] Enhanced NLP for token classification
- [ ] Plugin system for custom experiments
- [ ] Integration with more model libraries
- [ ] Export functionality for experiment results
- [ ] Agent collaboration workflows
- [ ] Documentation website

## 📄 License

See individual project files for licensing information. New unified components follow the same licensing as the original project.