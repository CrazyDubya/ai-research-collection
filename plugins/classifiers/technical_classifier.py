"""
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
