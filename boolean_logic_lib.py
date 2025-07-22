"""
Shared Boolean Logic Library
Consolidates boolean gate logic and token classification functionality.
"""

from typing import Dict, List, Callable, Tuple, Any, Optional
import random
import logging

# Optional imports for ML functionality
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy numpy for basic functionality
    class DummyNumpy:
        @staticmethod
        def array(x):
            return x
    np = DummyNumpy()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pure Boolean Gate Definitions
PURE_GATES = {
    "AND": (2, lambda a, b: a and b),
    "OR": (2, lambda a, b: a or b),
    "NAND": (2, lambda a, b: not (a and b)),
    "NOR": (2, lambda a, b: not (a or b)),
    "XOR": (2, lambda a, b: (a and not b) or (not a and b)),
    "XNOR": (2, lambda a, b: (a and b) or (not a and not b)),
    "NOT": (1, lambda a: not a)
}

# Default classification dictionaries (can be overridden via config)
DEFAULT_LOGICAL_WORDS = [
    "therefore", "thus", "hence", "consequently", "because", "since", "given",
    "analysis", "data", "evidence", "proof", "logic", "reasoning", "conclusion",
    "study", "research", "method", "approach", "theory", "hypothesis", "result"
]

DEFAULT_EMOTIONAL_WORDS = [
    "love", "hate", "fear", "joy", "anger", "sadness", "excitement", "passion",
    "heart", "soul", "feeling", "emotion", "mood", "spirit", "warm", "cold"
]

DEFAULT_LOGICAL_PHRASES = [
    "in conclusion", "as a result", "it follows that", "based on evidence",
    "the data shows", "research indicates", "studies suggest", "analysis reveals"
]

DEFAULT_EMOTIONAL_PHRASES = [
    "i feel", "my heart", "deep down", "emotional connection", "gut feeling",
    "from the heart", "touches my soul", "makes me feel"
]


class BooleanGateLibrary:
    """Centralized boolean gate operations with consistent API."""
    
    def __init__(self):
        self.gates = PURE_GATES.copy()
    
    def evaluate_gate(self, gate_name: str, *inputs) -> bool:
        """Evaluate a boolean gate with given inputs."""
        if gate_name not in self.gates:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        arity, func = self.gates[gate_name]
        if len(inputs) != arity:
            raise ValueError(f"Gate {gate_name} expects {arity} inputs, got {len(inputs)}")
        
        return func(*inputs)
    
    def get_gate_arity(self, gate_name: str) -> int:
        """Get the number of inputs required for a gate."""
        if gate_name not in self.gates:
            raise ValueError(f"Unknown gate: {gate_name}")
        return self.gates[gate_name][0]
    
    def list_gates(self) -> List[str]:
        """List all available gates."""
        return list(self.gates.keys())
    
    def add_custom_gate(self, name: str, arity: int, func: Callable) -> None:
        """Add a custom gate definition."""
        self.gates[name] = (arity, func)
        logger.info(f"Added custom gate: {name} with arity {arity}")


class TokenClassifier:
    """Abstract token classification with configurable dictionaries."""
    
    def __init__(self, 
                 logical_words: Optional[List[str]] = None,
                 emotional_words: Optional[List[str]] = None,
                 logical_phrases: Optional[List[str]] = None,
                 emotional_phrases: Optional[List[str]] = None):
        self.logical_words = logical_words or DEFAULT_LOGICAL_WORDS.copy()
        self.emotional_words = emotional_words or DEFAULT_EMOTIONAL_WORDS.copy()
        self.logical_phrases = logical_phrases or DEFAULT_LOGICAL_PHRASES.copy()
        self.emotional_phrases = emotional_phrases or DEFAULT_EMOTIONAL_PHRASES.copy()
    
    def classify_token(self, token: str, text_context: str = "") -> str:
        """Classify a token as 'logical', 'emotional', or 'neutral'."""
        token_lower = token.lower()
        text_lower = text_context.lower()
        
        # Check phrases first (more specific)
        for phrase in self.logical_phrases:
            if phrase in text_lower:
                return "logical"
        
        for phrase in self.emotional_phrases:
            if phrase in text_lower:
                return "emotional"
        
        # Check individual words
        if token_lower in self.logical_words:
            return "logical"
        elif token_lower in self.emotional_words:
            return "emotional"
        
        return "neutral"
    
    def classify_tokens_batch(self, tokens: List[str], text_context: str = "") -> List[str]:
        """Classify a batch of tokens."""
        return [self.classify_token(token, text_context) for token in tokens]
    
    def update_dictionaries(self, 
                          logical_words: Optional[List[str]] = None,
                          emotional_words: Optional[List[str]] = None,
                          logical_phrases: Optional[List[str]] = None,
                          emotional_phrases: Optional[List[str]] = None) -> None:
        """Update classification dictionaries at runtime."""
        if logical_words:
            self.logical_words.extend(logical_words)
        if emotional_words:
            self.emotional_words.extend(emotional_words)
        if logical_phrases:
            self.logical_phrases.extend(logical_phrases)
        if emotional_phrases:
            self.emotional_phrases.extend(emotional_phrases)
        
        logger.info("Updated classification dictionaries")


class BooleanLogicExperiment:
    """Unified experiment runner for boolean logic with multiple models."""
    
    def __init__(self, gate_lib: Optional[BooleanGateLibrary] = None,
                 classifier: Optional[TokenClassifier] = None):
        self.gate_lib = gate_lib or BooleanGateLibrary()
        self.classifier = classifier or TokenClassifier()
        self.models = {}
        self.results = []
    
    def add_model(self, model_id: str, model, tokenizer):
        """Add a model to the experiment."""
        self.models[model_id] = {
            'model': model,
            'tokenizer': tokenizer
        }
        logger.info(f"Added model: {model_id}")
    
    def remove_model(self, model_id: str):
        """Remove a model from the experiment."""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Removed model: {model_id}")
    
    def run_gate_experiment(self, gate_name: str, input_text: str, 
                          logical_input: bool, emotional_input: bool) -> Dict:
        """Run a boolean gate experiment on input text."""
        gate_result = self.gate_lib.evaluate_gate(gate_name, logical_input, emotional_input)
        
        results = {
            'gate': gate_name,
            'logical_input': logical_input,
            'emotional_input': emotional_input,
            'gate_result': gate_result,
            'input_text': input_text,
            'model_results': {}
        }
        
        # Process with each model
        for model_id, model_data in self.models.items():
            try:
                tokenizer = model_data['tokenizer']
                tokens = tokenizer.tokenize(input_text)
                classifications = self.classifier.classify_tokens_batch(tokens, input_text)
                
                results['model_results'][model_id] = {
                    'tokens': tokens,
                    'classifications': classifications,
                    'logical_count': classifications.count('logical'),
                    'emotional_count': classifications.count('emotional'),
                    'neutral_count': classifications.count('neutral')
                }
            except Exception as e:
                logger.error(f"Error processing model {model_id}: {e}")
                results['model_results'][model_id] = {'error': str(e)}
        
        self.results.append(results)
        return results


def generate_truth_tables() -> str:
    """Generate markdown truth tables for all boolean gates."""
    output = []
    gate_lib = BooleanGateLibrary()
    
    for name in gate_lib.list_gates():
        arity = gate_lib.get_gate_arity(name)
        
        if arity == 2:
            output.append(f"### {name} Gate Truth Table")
            output.append("| A | B | {} |".format(name))
            output.append("|:-:|:-:|:-:|")
            for A in (0, 1):
                for B in (0, 1):
                    result = int(gate_lib.evaluate_gate(name, A, B))
                    output.append(f"| {A} | {B} | {result} |")
            output.append("")
        else:  # arity == 1
            output.append(f"### {name} Gate Truth Table")
            output.append("| A | {} |".format(name))
            output.append("|:-:|:-:|")
            for A in (0, 1):
                result = int(gate_lib.evaluate_gate(name, A))
                output.append(f"| {A} | {result} |")
            output.append("")
    
    return "\n".join(output)


# Convenience functions for backward compatibility
def and_gate(a: bool, b: bool) -> bool:
    """Boolean AND gate."""
    return PURE_GATES["AND"][1](a, b)

def or_gate(a: bool, b: bool) -> bool:
    """Boolean OR gate."""
    return PURE_GATES["OR"][1](a, b)

def xor_gate(a: bool, b: bool) -> bool:
    """Boolean XOR gate."""
    return PURE_GATES["XOR"][1](a, b)

def not_gate(a: bool) -> bool:
    """Boolean NOT gate."""
    return PURE_GATES["NOT"][1](a)