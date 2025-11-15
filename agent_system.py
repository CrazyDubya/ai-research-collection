"""
Unified Agent Interface for AI Research Collection
Standardizes agent faculties and provides consistent API for agent interactions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AgentFaculty(Enum):
    """Standard agent faculties."""
    RECALL = "RECALL"
    CONSULT = "CONSULT"
    THINK = "THINK"
    TALK = "TALK"
    DONE = "DONE"

class AgentAction:
    """Represents an agent action with metadata."""
    
    def __init__(self, faculty: AgentFaculty, content: str, metadata: Optional[Dict] = None):
        self.faculty = faculty
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.id = f"{faculty.value}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            'id': self.id,
            'faculty': self.faculty.value,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentAction':
        """Create action from dictionary."""
        action = cls(
            AgentFaculty(data['faculty']),
            data['content'],
            data.get('metadata', {})
        )
        action.id = data['id']
        action.timestamp = datetime.fromisoformat(data['timestamp'])
        return action

class AgentMemory:
    """Agent memory management with recall capabilities."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.memories: List[Dict[str, Any]] = []
        self.semantic_index: Dict[str, List[int]] = {}
    
    def add_memory(self, content: str, memory_type: str = "general", tags: Optional[List[str]] = None) -> None:
        """Add a memory to the agent's memory store."""
        memory = {
            'id': len(self.memories),
            'content': content,
            'type': memory_type,
            'tags': tags or [],
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
        
        self.memories.append(memory)
        
        # Update semantic index
        keywords = self._extract_keywords(content)
        for keyword in keywords:
            if keyword not in self.semantic_index:
                self.semantic_index[keyword] = []
            self.semantic_index[keyword].append(memory['id'])
        
        # Maintain max size
        if len(self.memories) > self.max_size:
            self._forget_oldest()
        
        logger.debug(f"Added memory: {content[:50]}...")
    
    def recall(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Recall memories based on a query."""
        query_keywords = self._extract_keywords(query.lower())
        memory_scores: Dict[int, float] = {}
        
        # Score memories based on keyword matches
        for keyword in query_keywords:
            if keyword in self.semantic_index:
                for memory_id in self.semantic_index[keyword]:
                    if memory_id < len(self.memories):  # Ensure memory still exists
                        if memory_id not in memory_scores:
                            memory_scores[memory_id] = 0
                        memory_scores[memory_id] += 1
        
        # Sort by score and recency
        sorted_memories = sorted(
            memory_scores.items(),
            key=lambda x: (x[1], -x[0]),  # Score descending, recency descending
            reverse=True
        )
        
        # Update access count and return top results
        results = []
        for memory_id, score in sorted_memories[:max_results]:
            memory = self.memories[memory_id]
            memory['access_count'] += 1
            results.append(memory.copy())
        
        logger.debug(f"Recalled {len(results)} memories for query: {query}")
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for semantic indexing."""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word.strip('.,!?;:"()[]') for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _forget_oldest(self) -> None:
        """Remove oldest memory to maintain size limit."""
        if self.memories:
            oldest_memory = self.memories.pop(0)
            # Update semantic index
            keywords = self._extract_keywords(oldest_memory['content'])
            for keyword in keywords:
                if keyword in self.semantic_index:
                    self.semantic_index[keyword] = [
                        mid for mid in self.semantic_index[keyword] 
                        if mid != oldest_memory['id']
                    ]
                    if not self.semantic_index[keyword]:
                        del self.semantic_index[keyword]

class BaseAgent(ABC):
    """Base class for all agents with standardized interface."""
    
    def __init__(self, name: str, max_iterations: int = 10, memory_size: int = 1000):
        self.name = name
        self.max_iterations = max_iterations
        self.memory = AgentMemory(memory_size)
        self.action_history: List[AgentAction] = []
        self.current_iteration = 0
        self.is_done = False
        self.supported_faculties = [AgentFaculty.RECALL, AgentFaculty.THINK, AgentFaculty.TALK, AgentFaculty.DONE]
        
        # Initialize with basic knowledge
        self._initialize_memory()
    
    @abstractmethod
    def _initialize_memory(self) -> None:
        """Initialize agent with basic knowledge/memories."""
        pass
    
    @abstractmethod
    def process_input(self, input_text: str) -> AgentAction:
        """Process input and return the next action."""
        pass
    
    def execute_faculty(self, faculty: AgentFaculty, content: str) -> AgentAction:
        """Execute a specific faculty action."""
        if faculty not in self.supported_faculties:
            raise ValueError(f"Faculty {faculty} not supported by {self.name}")
        
        action = AgentAction(faculty, content)
        self.action_history.append(action)
        
        if faculty == AgentFaculty.RECALL:
            return self._execute_recall(content)
        elif faculty == AgentFaculty.CONSULT:
            return self._execute_consult(content)
        elif faculty == AgentFaculty.THINK:
            return self._execute_think(content)
        elif faculty == AgentFaculty.TALK:
            return self._execute_talk(content)
        elif faculty == AgentFaculty.DONE:
            self.is_done = True
            return action
        
        return action
    
    def _execute_recall(self, query: str) -> AgentAction:
        """Execute RECALL faculty."""
        memories = self.memory.recall(query)
        if memories:
            recalled_content = "\n".join([f"- {mem['content']}" for mem in memories[:3]])
            result = f"Recalled memories for '{query}':\n{recalled_content}"
        else:
            result = f"No relevant memories found for '{query}'"
        
        return AgentAction(AgentFaculty.RECALL, result, {'query': query, 'memories_found': len(memories)})
    
    def _execute_consult(self, document_name: str) -> AgentAction:
        """Execute CONSULT faculty."""
        # Default implementation - can be overridden
        result = f"Consulted document: {document_name} (implementation specific to agent type)"
        return AgentAction(AgentFaculty.CONSULT, result, {'document': document_name})
    
    def _execute_think(self, thought: str) -> AgentAction:
        """Execute THINK faculty."""
        # Add thought to memory for future recall
        self.memory.add_memory(thought, "thought", ["thinking", "internal"])
        result = f"Thinking: {thought}"
        return AgentAction(AgentFaculty.THINK, result, {'thought_added_to_memory': True})
    
    def _execute_talk(self, response: str) -> AgentAction:
        """Execute TALK faculty."""
        # Add response to memory
        self.memory.add_memory(response, "communication", ["talking", "output"])
        return AgentAction(AgentFaculty.TALK, response, {'response_type': 'talk'})
    
    def add_memory(self, content: str, memory_type: str = "general", tags: Optional[List[str]] = None) -> None:
        """Add a memory to the agent."""
        self.memory.add_memory(content, memory_type, tags)
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get the agent's action history."""
        return [action.to_dict() for action in self.action_history]
    
    def reset(self) -> None:
        """Reset agent state."""
        self.current_iteration = 0
        self.is_done = False
        self.action_history.clear()
        logger.info(f"Agent {self.name} reset")

class LiteratureAgent(BaseAgent):
    """Specialized agent for literature analysis tasks."""
    
    def __init__(self, name: str = "LitAgent", **kwargs):
        super().__init__(name, **kwargs)
        self.supported_faculties.append(AgentFaculty.CONSULT)  # Add CONSULT support
        self.documents: Dict[str, str] = {}
    
    def _initialize_memory(self) -> None:
        """Initialize with literature analysis knowledge."""
        knowledge_base = [
            "Literature analysis involves examining themes, characters, style, and context",
            "Common literary devices include metaphor, symbolism, irony, and foreshadowing",
            "Character development can be analyzed through actions, dialogue, and internal thoughts",
            "Historical and cultural context influences literary interpretation",
            "Genre conventions affect reader expectations and interpretation"
        ]
        
        for knowledge in knowledge_base:
            self.memory.add_memory(knowledge, "knowledge", ["literature", "analysis"])
    
    def add_document(self, name: str, content: str) -> None:
        """Add a document for consultation."""
        self.documents[name] = content
        self.memory.add_memory(f"Document available: {name}", "document", ["consultation", "available"])
        logger.info(f"Added document: {name}")
    
    def _execute_consult(self, document_name: str) -> AgentAction:
        """Execute CONSULT faculty for literature documents."""
        if document_name in self.documents:
            content = self.documents[document_name]
            # Add document content to memory for analysis
            self.memory.add_memory(f"Consulted {document_name}: {content[:200]}...", "consultation", ["document", document_name])
            result = f"Consulted '{document_name}' - content available for analysis"
            metadata = {'document_found': True, 'content_length': len(content)}
        else:
            available_docs = list(self.documents.keys())
            result = f"Document '{document_name}' not found. Available: {available_docs}"
            metadata = {'document_found': False, 'available_documents': available_docs}
        
        return AgentAction(AgentFaculty.CONSULT, result, metadata)
    
    def process_input(self, input_text: str) -> AgentAction:
        """Process input for literature analysis."""
        input_lower = input_text.lower()
        
        # Simple decision logic - could be enhanced with LLM
        if "analyze" in input_lower or "analysis" in input_lower:
            return self.execute_faculty(AgentFaculty.THINK, f"Analyzing: {input_text}")
        elif "recall" in input_lower or "remember" in input_lower:
            query = input_text.replace("recall", "").replace("remember", "").strip()
            return self.execute_faculty(AgentFaculty.RECALL, query)
        elif "consult" in input_lower or "document" in input_lower:
            doc_name = input_text.split()[-1]  # Simple extraction
            return self.execute_faculty(AgentFaculty.CONSULT, doc_name)
        elif "done" in input_lower or "finished" in input_lower:
            return self.execute_faculty(AgentFaculty.DONE, "Analysis complete")
        else:
            return self.execute_faculty(AgentFaculty.TALK, f"I understand you want me to consider: {input_text}")

class BooleanLogicAgent(BaseAgent):
    """Specialized agent for boolean logic experiments."""
    
    def __init__(self, name: str = "BooleanAgent", **kwargs):
        super().__init__(name, **kwargs)
        self.gate_lib = None  # Will be injected
    
    def _initialize_memory(self) -> None:
        """Initialize with boolean logic knowledge."""
        knowledge_base = [
            "Boolean gates perform logical operations on binary inputs",
            "AND gate returns true only when both inputs are true",
            "OR gate returns true when at least one input is true",
            "XOR gate returns true when inputs are different",
            "NOT gate inverts the input",
            "Truth tables show all possible input/output combinations"
        ]
        
        for knowledge in knowledge_base:
            self.memory.add_memory(knowledge, "knowledge", ["boolean", "logic", "gates"])
    
    def set_gate_library(self, gate_lib) -> None:
        """Inject boolean gate library."""
        self.gate_lib = gate_lib
        self.memory.add_memory("Boolean gate library connected", "system", ["gates", "ready"])
    
    def process_input(self, input_text: str) -> AgentAction:
        """Process input for boolean logic tasks."""
        input_lower = input_text.lower()
        
        if "gate" in input_lower and self.gate_lib:
            gates = self.gate_lib.list_gates()
            return self.execute_faculty(AgentFaculty.TALK, f"Available gates: {', '.join(gates)}")
        elif "truth table" in input_lower:
            return self.execute_faculty(AgentFaculty.THINK, "Generating truth tables for boolean gates")
        elif "experiment" in input_lower:
            return self.execute_faculty(AgentFaculty.THINK, f"Planning boolean logic experiment: {input_text}")
        elif "done" in input_lower:
            return self.execute_faculty(AgentFaculty.DONE, "Boolean logic analysis complete")
        else:
            return self.execute_faculty(AgentFaculty.RECALL, input_text)

class AgentOrchestrator:
    """Manages multiple agents and coordinates their interactions."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.interaction_history: List[Dict[str, Any]] = []
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the orchestrator."""
        self.agents[agent.name] = agent
        logger.info(f"Added agent: {agent.name}")
    
    def remove_agent(self, name: str) -> None:
        """Remove an agent."""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Removed agent: {name}")
    
    def run_agent_interaction(self, agent_name: str, input_text: str) -> Optional[AgentAction]:
        """Run a single interaction with an agent."""
        if agent_name not in self.agents:
            logger.error(f"Agent {agent_name} not found")
            return None
        
        agent = self.agents[agent_name]
        if agent.is_done:
            logger.info(f"Agent {agent_name} is already done")
            return None
        
        action = agent.process_input(input_text)
        
        # Record interaction
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'input': input_text,
            'action': action.to_dict()
        }
        self.interaction_history.append(interaction)
        
        return action
    
    def run_multi_agent_conversation(self, participants: List[str], topic: str, max_rounds: int = 5) -> List[Dict[str, Any]]:
        """Run a conversation between multiple agents."""
        conversation = []
        current_topic = topic
        
        for round_num in range(max_rounds):
            round_actions = []
            
            for agent_name in participants:
                if agent_name in self.agents and not self.agents[agent_name].is_done:
                    action = self.run_agent_interaction(agent_name, current_topic)
                    if action:
                        round_actions.append({
                            'agent': agent_name,
                            'action': action.to_dict()
                        })
                        
                        # Update topic based on agent response
                        if action.faculty == AgentFaculty.TALK:
                            current_topic = action.content
            
            if round_actions:
                conversation.append({
                    'round': round_num + 1,
                    'actions': round_actions
                })
            
            # Check if all agents are done
            if all(self.agents[name].is_done for name in participants if name in self.agents):
                break
        
        return conversation
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents."""
        status = {}
        for name, agent in self.agents.items():
            status[name] = {
                'is_done': agent.is_done,
                'current_iteration': agent.current_iteration,
                'action_count': len(agent.action_history),
                'memory_count': len(agent.memory.memories),
                'supported_faculties': [f.value for f in agent.supported_faculties]
            }
        return status

# Convenience functions for creating standard agents
def create_literature_agent(name: str = "LitAgent") -> LiteratureAgent:
    """Create a standard literature analysis agent."""
    return LiteratureAgent(name)

def create_boolean_logic_agent(name: str = "BooleanAgent") -> BooleanLogicAgent:
    """Create a standard boolean logic agent."""
    return BooleanLogicAgent(name)