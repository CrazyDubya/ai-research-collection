# AI Research & Development Collection

A collection of AI research projects, experiments, and utilities focusing on multi-model analysis, agent simulation, and document generation.

## Projects Overview

### Boolean Logic Analysis
- **boolean_logic_multimodel.py** - Multi-model boolean logic evaluation
- **boolean_gates.py** - Boolean gate implementations and testing
- **boolean_logic_larger_model.py** - Extended boolean logic analysis

### Character & Agent Systems
- **chars/** - Character generation and management
  - Fantasy character builders
  - Political and court character generators
- **hermes/** - Advanced character building system
- **void/** - Simulation engine experiments

### Document Generation
- **synth-hosts.py** - Synthetic host generation system  
- **lisa_diary*.py** - Personal diary generation experiments
- **syn-gen.py** - General synthesis generator

### Web Applications
- **chat_serv/** - Real-time chat server (Node.js)
- **hackathon.py** - Web content generation tools

### Analysis & Utilities
- **LitAg.py** - Literature analysis agent
- **memory.py** - Memory management utilities
- **sanitize/** - Text sanitization and benchmarking tools

### Research Experiments
- **gpu_math.py** - GPU-accelerated mathematical operations
- **monte.py** - Monte Carlo simulations
- **novel-math.py** - Novel mathematical approaches
- **space.py** - Spatial analysis tools

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies (if requirements.txt exists):
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

Each Python script can typically be run independently:
```bash
python script_name.py
```

For specific usage instructions, check the individual script documentation or run with `--help` flag where available.

## Project Structure

- **Core Scripts**: Main Python files in root directory
- **chars/**: Character generation modules
- **chat_serv/**: Web server components  
- **hermes/**: Advanced character systems
- **sanitize/**: Text processing utilities
- **void/**: Experimental simulation engines

## Contributing

This is a research collection - feel free to experiment and extend any components.

## License

See individual project files for licensing information.