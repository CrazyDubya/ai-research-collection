"""
Plugin architecture for AI Research Collection
Demonstrates how to extend the system with custom components.
"""

import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Type, Protocol
import logging

logger = logging.getLogger(__name__)

class GatePlugin(Protocol):
    """Protocol for custom gate plugins."""
    
    def get_gate_name(self) -> str:
        """Return the name of the gate."""
        ...
    
    def get_gate_arity(self) -> int:
        """Return the number of inputs the gate accepts."""
        ...
    
    def evaluate(self, *inputs) -> bool:
        """Evaluate the gate with given inputs."""
        ...

class AgentPlugin(Protocol):
    """Protocol for custom agent plugins."""
    
    def get_agent_type(self) -> str:
        """Return the type name of the agent."""
        ...
    
    def create_agent(self, name: str, **kwargs) -> Any:
        """Create an instance of the agent."""
        ...

class ClassifierPlugin(Protocol):
    """Protocol for custom classification plugins."""
    
    def get_classifier_name(self) -> str:
        """Return the name of the classifier."""
        ...
    
    def classify_token(self, token: str, context: str = "") -> str:
        """Classify a token."""
        ...

class PluginManager:
    """Manages loading and registration of plugins."""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        
        self.gate_plugins: Dict[str, GatePlugin] = {}
        self.agent_plugins: Dict[str, AgentPlugin] = {}
        self.classifier_plugins: Dict[str, ClassifierPlugin] = {}
        
        # Create plugin directory structure
        (self.plugin_dir / "gates").mkdir(exist_ok=True)
        (self.plugin_dir / "agents").mkdir(exist_ok=True)
        (self.plugin_dir / "classifiers").mkdir(exist_ok=True)
        
        # Create __init__.py files
        for subdir in ["gates", "agents", "classifiers"]:
            init_file = self.plugin_dir / subdir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    def load_plugins(self) -> None:
        """Load all plugins from the plugin directory."""
        # Add plugin directory to Python path
        if str(self.plugin_dir) not in sys.path:
            sys.path.insert(0, str(self.plugin_dir))
        
        self._load_gate_plugins()
        self._load_agent_plugins()
        self._load_classifier_plugins()
        
        logger.info(f"Loaded {len(self.gate_plugins)} gate plugins, "
                   f"{len(self.agent_plugins)} agent plugins, "
                   f"{len(self.classifier_plugins)} classifier plugins")
    
    def _load_gate_plugins(self) -> None:
        """Load gate plugins."""
        gates_dir = self.plugin_dir / "gates"
        for plugin_file in gates_dir.glob("*.py"):
            if plugin_file.name == "__init__.py":
                continue
            
            try:
                module_name = f"gates.{plugin_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for classes that implement GatePlugin
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        hasattr(attr, 'get_gate_name') and 
                        hasattr(attr, 'evaluate')):
                        
                        plugin_instance = attr()
                        gate_name = plugin_instance.get_gate_name()
                        self.gate_plugins[gate_name] = plugin_instance
                        logger.info(f"Loaded gate plugin: {gate_name}")
                        
            except Exception as e:
                logger.error(f"Failed to load gate plugin {plugin_file}: {e}")
    
    def _load_agent_plugins(self) -> None:
        """Load agent plugins."""
        agents_dir = self.plugin_dir / "agents"
        for plugin_file in agents_dir.glob("*.py"):
            if plugin_file.name == "__init__.py":
                continue
            
            try:
                module_name = f"agents.{plugin_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for classes that implement AgentPlugin
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        hasattr(attr, 'get_agent_type') and 
                        hasattr(attr, 'create_agent')):
                        
                        plugin_instance = attr()
                        agent_type = plugin_instance.get_agent_type()
                        self.agent_plugins[agent_type] = plugin_instance
                        logger.info(f"Loaded agent plugin: {agent_type}")
                        
            except Exception as e:
                logger.error(f"Failed to load agent plugin {plugin_file}: {e}")
    
    def _load_classifier_plugins(self) -> None:
        """Load classifier plugins."""
        classifiers_dir = self.plugin_dir / "classifiers"
        for plugin_file in classifiers_dir.glob("*.py"):
            if plugin_file.name == "__init__.py":
                continue
            
            try:
                module_name = f"classifiers.{plugin_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for classes that implement ClassifierPlugin
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        hasattr(attr, 'get_classifier_name') and 
                        hasattr(attr, 'classify_token')):
                        
                        plugin_instance = attr()
                        classifier_name = plugin_instance.get_classifier_name()
                        self.classifier_plugins[classifier_name] = plugin_instance
                        logger.info(f"Loaded classifier plugin: {classifier_name}")
                        
            except Exception as e:
                logger.error(f"Failed to load classifier plugin {plugin_file}: {e}")
    
    def get_gate_plugin(self, name: str) -> GatePlugin:
        """Get a gate plugin by name."""
        if name not in self.gate_plugins:
            raise ValueError(f"Gate plugin '{name}' not found")
        return self.gate_plugins[name]
    
    def get_agent_plugin(self, agent_type: str) -> AgentPlugin:
        """Get an agent plugin by type."""
        if agent_type not in self.agent_plugins:
            raise ValueError(f"Agent plugin '{agent_type}' not found")
        return self.agent_plugins[agent_type]
    
    def get_classifier_plugin(self, name: str) -> ClassifierPlugin:
        """Get a classifier plugin by name."""
        if name not in self.classifier_plugins:
            raise ValueError(f"Classifier plugin '{name}' not found")
        return self.classifier_plugins[name]
    
    def list_plugins(self) -> Dict[str, List[str]]:
        """List all available plugins."""
        return {
            'gates': list(self.gate_plugins.keys()),
            'agents': list(self.agent_plugins.keys()),
            'classifiers': list(self.classifier_plugins.keys())
        }
    
    def create_example_plugins(self) -> None:
        """Create example plugins for demonstration."""
        # Example gate plugin
        gate_example = '''"""
Example custom gate plugin - IMPLIES gate
"""

class ImpliesGate:
    """Logical implication gate: A IMPLIES B = (NOT A) OR B"""
    
    def get_gate_name(self) -> str:
        return "IMPLIES"
    
    def get_gate_arity(self) -> int:
        return 2
    
    def evaluate(self, a: bool, b: bool) -> bool:
        return (not a) or b
'''
        
        gate_file = self.plugin_dir / "gates" / "implies_gate.py"
        if not gate_file.exists():
            gate_file.write_text(gate_example)
        
        # Example classifier plugin
        classifier_example = '''"""
Example custom classifier plugin - Technical term classifier
"""

class TechnicalClassifier:
    """Classifies technical/scientific terms"""
    
    def __init__(self):
        self.technical_terms = [
            "algorithm", "neural", "network", "dataset", "model",
            "training", "validation", "optimization", "gradient",
            "tensor", "matrix", "vector", "computation"
        ]
    
    def get_classifier_name(self) -> str:
        return "technical"
    
    def classify_token(self, token: str, context: str = "") -> str:
        if token.lower() in self.technical_terms:
            return "technical"
        return "non_technical"
'''
        
        classifier_file = self.plugin_dir / "classifiers" / "technical_classifier.py"
        if not classifier_file.exists():
            classifier_file.write_text(classifier_example)
        
        # Example agent plugin
        agent_example = '''"""
Example custom agent plugin - Math agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent_system import BaseAgent, AgentFaculty, AgentAction

class MathAgent(BaseAgent):
    """Agent specialized in mathematical reasoning"""
    
    def _initialize_memory(self):
        math_knowledge = [
            "Mathematics is the study of patterns, structures, and relationships",
            "Algebra deals with symbols and rules for manipulating symbols",
            "Calculus studies continuous change and motion",
            "Statistics analyzes and interprets data",
            "Geometry studies shapes, sizes, and properties of space"
        ]
        
        for knowledge in math_knowledge:
            self.memory.add_memory(knowledge, "knowledge", ["mathematics", "theory"])
    
    def process_input(self, input_text: str) -> AgentAction:
        input_lower = input_text.lower()
        
        if any(math_word in input_lower for math_word in ["calculate", "solve", "equation", "formula"]):
            return self.execute_faculty(AgentFaculty.THINK, f"Analyzing mathematical problem: {input_text}")
        elif "recall" in input_lower:
            query = input_text.replace("recall", "").strip()
            return self.execute_faculty(AgentFaculty.RECALL, query)
        elif "done" in input_lower:
            return self.execute_faculty(AgentFaculty.DONE, "Mathematical analysis complete")
        else:
            return self.execute_faculty(AgentFaculty.TALK, f"I can help with mathematical reasoning: {input_text}")

class MathAgentPlugin:
    """Plugin wrapper for MathAgent"""
    
    def get_agent_type(self) -> str:
        return "math"
    
    def create_agent(self, name: str, **kwargs):
        return MathAgent(name, **kwargs)
'''
        
        agent_file = self.plugin_dir / "agents" / "math_agent.py"
        if not agent_file.exists():
            agent_file.write_text(agent_example)
        
        logger.info("Created example plugins")


# Global plugin manager instance
plugin_manager = PluginManager()

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    return plugin_manager

def load_plugins() -> None:
    """Load all plugins."""
    plugin_manager.load_plugins()

def create_example_plugins() -> None:
    """Create example plugins."""
    plugin_manager.create_example_plugins()


# Integration with existing systems
def integrate_gate_plugins(gate_library):
    """Integrate gate plugins with BooleanGateLibrary."""
    for name, plugin in plugin_manager.gate_plugins.items():
        gate_library.add_custom_gate(
            name, 
            plugin.get_gate_arity(), 
            plugin.evaluate
        )
        logger.info(f"Integrated gate plugin: {name}")

def integrate_agent_plugins(orchestrator):
    """Integrate agent plugins with AgentOrchestrator."""
    # This would require extending the orchestrator to support plugin-created agents
    # For now, just log available plugins
    for agent_type in plugin_manager.agent_plugins.keys():
        logger.info(f"Available agent plugin: {agent_type}")

if __name__ == "__main__":
    # Demo the plugin system
    print("AI Research Collection Plugin System Demo")
    print("=" * 50)
    
    # Create example plugins
    create_example_plugins()
    print("Created example plugins")
    
    # Load plugins
    load_plugins()
    
    # List available plugins
    plugins = plugin_manager.list_plugins()
    print(f"\\nAvailable plugins:")
    for category, plugin_list in plugins.items():
        print(f"  {category}: {plugin_list}")
    
    # Test gate plugin
    if "IMPLIES" in plugin_manager.gate_plugins:
        implies_gate = plugin_manager.get_gate_plugin("IMPLIES")
        result = implies_gate.evaluate(True, False)
        print(f"\\nTesting IMPLIES gate: True IMPLIES False = {result}")
    
    # Test classifier plugin
    if "technical" in plugin_manager.classifier_plugins:
        tech_classifier = plugin_manager.get_classifier_plugin("technical")
        result = tech_classifier.classify_token("algorithm")
        print(f"Testing technical classifier: 'algorithm' = {result}")
    
    print("\\nPlugin system demo complete!")