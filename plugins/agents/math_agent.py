"""
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
