"""
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
