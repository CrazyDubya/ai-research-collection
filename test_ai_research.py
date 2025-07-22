#!/usr/bin/env python3
"""
Test suite for AI Research Collection refactored components
Basic tests for boolean logic library and configuration management.
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boolean_logic_lib import BooleanGateLibrary, TokenClassifier, BooleanLogicExperiment
from config_manager import ConfigManager


class TestBooleanGateLibrary(unittest.TestCase):
    """Test the centralized boolean gate library."""
    
    def setUp(self):
        self.gate_lib = BooleanGateLibrary()
    
    def test_and_gate(self):
        """Test AND gate functionality."""
        self.assertTrue(self.gate_lib.evaluate_gate("AND", True, True))
        self.assertFalse(self.gate_lib.evaluate_gate("AND", True, False))
        self.assertFalse(self.gate_lib.evaluate_gate("AND", False, True))
        self.assertFalse(self.gate_lib.evaluate_gate("AND", False, False))
    
    def test_or_gate(self):
        """Test OR gate functionality."""
        self.assertTrue(self.gate_lib.evaluate_gate("OR", True, True))
        self.assertTrue(self.gate_lib.evaluate_gate("OR", True, False))
        self.assertTrue(self.gate_lib.evaluate_gate("OR", False, True))
        self.assertFalse(self.gate_lib.evaluate_gate("OR", False, False))
    
    def test_xor_gate(self):
        """Test XOR gate functionality."""
        self.assertFalse(self.gate_lib.evaluate_gate("XOR", True, True))
        self.assertTrue(self.gate_lib.evaluate_gate("XOR", True, False))
        self.assertTrue(self.gate_lib.evaluate_gate("XOR", False, True))
        self.assertFalse(self.gate_lib.evaluate_gate("XOR", False, False))
    
    def test_not_gate(self):
        """Test NOT gate functionality."""
        self.assertFalse(self.gate_lib.evaluate_gate("NOT", True))
        self.assertTrue(self.gate_lib.evaluate_gate("NOT", False))
    
    def test_gate_arity(self):
        """Test gate arity checking."""
        self.assertEqual(self.gate_lib.get_gate_arity("AND"), 2)
        self.assertEqual(self.gate_lib.get_gate_arity("NOT"), 1)
    
    def test_invalid_gate(self):
        """Test error handling for invalid gates."""
        with self.assertRaises(ValueError):
            self.gate_lib.evaluate_gate("INVALID", True, True)
    
    def test_wrong_arity(self):
        """Test error handling for wrong number of inputs."""
        with self.assertRaises(ValueError):
            self.gate_lib.evaluate_gate("AND", True)  # Needs 2 inputs
        
        with self.assertRaises(ValueError):
            self.gate_lib.evaluate_gate("NOT", True, False)  # Needs 1 input
    
    def test_custom_gate(self):
        """Test adding custom gates."""
        # Add a simple NAND gate manually
        self.gate_lib.add_custom_gate("CUSTOM_NAND", 2, lambda a, b: not (a and b))
        
        self.assertTrue(self.gate_lib.evaluate_gate("CUSTOM_NAND", False, False))
        self.assertTrue(self.gate_lib.evaluate_gate("CUSTOM_NAND", True, False))
        self.assertFalse(self.gate_lib.evaluate_gate("CUSTOM_NAND", True, True))
    
    def test_list_gates(self):
        """Test listing available gates."""
        gates = self.gate_lib.list_gates()
        expected_gates = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR", "NOT"]
        for gate in expected_gates:
            self.assertIn(gate, gates)


class TestTokenClassifier(unittest.TestCase):
    """Test the token classifier."""
    
    def setUp(self):
        self.classifier = TokenClassifier()
    
    def test_logical_classification(self):
        """Test classification of logical words."""
        self.assertEqual(self.classifier.classify_token("therefore"), "logical")
        self.assertEqual(self.classifier.classify_token("analysis"), "logical")
        self.assertEqual(self.classifier.classify_token("evidence"), "logical")
    
    def test_emotional_classification(self):
        """Test classification of emotional words."""
        self.assertEqual(self.classifier.classify_token("love"), "emotional")
        self.assertEqual(self.classifier.classify_token("heart"), "emotional")
        self.assertEqual(self.classifier.classify_token("passion"), "emotional")
    
    def test_neutral_classification(self):
        """Test classification of neutral words."""
        self.assertEqual(self.classifier.classify_token("the"), "neutral")
        self.assertEqual(self.classifier.classify_token("chair"), "neutral")
        self.assertEqual(self.classifier.classify_token("random"), "neutral")
    
    def test_phrase_classification(self):
        """Test phrase-based classification."""
        text = "in conclusion, the data shows important results"
        self.assertEqual(self.classifier.classify_token("shows", text), "logical")
    
    def test_batch_classification(self):
        """Test batch token classification."""
        tokens = ["love", "therefore", "chair", "emotion"]
        classifications = self.classifier.classify_tokens_batch(tokens)
        expected = ["emotional", "logical", "neutral", "emotional"]
        self.assertEqual(classifications, expected)
    
    def test_case_insensitive(self):
        """Test case-insensitive classification."""
        self.assertEqual(self.classifier.classify_token("LOVE"), "emotional")
        self.assertEqual(self.classifier.classify_token("Therefore"), "logical")
    
    def test_custom_dictionaries(self):
        """Test custom classification dictionaries."""
        custom_logical = ["algorithm", "computation"]
        custom_emotional = ["empathy", "compassion"]
        
        classifier = TokenClassifier(
            logical_words=custom_logical,
            emotional_words=custom_emotional
        )
        
        self.assertEqual(classifier.classify_token("algorithm"), "logical")
        self.assertEqual(classifier.classify_token("empathy"), "emotional")
    
    def test_update_dictionaries(self):
        """Test runtime dictionary updates."""
        self.classifier.update_dictionaries(
            logical_words=["systematic"],
            emotional_words=["heartfelt"]
        )
        
        self.assertEqual(self.classifier.classify_token("systematic"), "logical")
        self.assertEqual(self.classifier.classify_token("heartfelt"), "emotional")


class TestConfigManager(unittest.TestCase):
    """Test the configuration manager."""
    
    def setUp(self):
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.test_dir)
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_default_config(self):
        """Test default configuration loading."""
        config = self.config_manager.config
        self.assertIn("boolean_logic", config)
        self.assertIn("models", config)
        self.assertIn("experiments", config)
        self.assertIn("agents", config)
    
    def test_get_config_value(self):
        """Test getting configuration values."""
        logical_words = self.config_manager.get("boolean_logic.logical_words")
        self.assertIsInstance(logical_words, list)
        self.assertIn("therefore", logical_words)
    
    def test_set_config_value(self):
        """Test setting configuration values."""
        self.config_manager.set("test.value", "hello")
        self.assertEqual(self.config_manager.get("test.value"), "hello")
    
    def test_add_to_list(self):
        """Test adding items to list configurations."""
        original_length = len(self.config_manager.get("boolean_logic.logical_words"))
        self.config_manager.add_to_list("boolean_logic.logical_words", "test_word")
        new_length = len(self.config_manager.get("boolean_logic.logical_words"))
        self.assertEqual(new_length, original_length + 1)
        self.assertIn("test_word", self.config_manager.get("boolean_logic.logical_words"))
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Modify config
        self.config_manager.set("test.save_load", "test_value")
        
        # Save config
        success = self.config_manager.save_config()
        self.assertTrue(success)
        
        # Create new config manager and load
        new_config_manager = ConfigManager(self.test_dir)
        new_config_manager.load_config()
        
        # Check if value was preserved
        self.assertEqual(new_config_manager.get("test.save_load"), "test_value")


class TestBooleanLogicExperiment(unittest.TestCase):
    """Test the experiment runner."""
    
    def setUp(self):
        self.gate_lib = BooleanGateLibrary()
        self.classifier = TokenClassifier()
        self.experiment = BooleanLogicExperiment(self.gate_lib, self.classifier)
    
    def test_experiment_initialization(self):
        """Test experiment initialization."""
        self.assertIsInstance(self.experiment.gate_lib, BooleanGateLibrary)
        self.assertIsInstance(self.experiment.classifier, TokenClassifier)
        self.assertEqual(len(self.experiment.models), 0)
        self.assertEqual(len(self.experiment.results), 0)
    
    def test_model_management(self):
        """Test adding and removing models."""
        # Mock model and tokenizer
        mock_model = "mock_model"
        mock_tokenizer = type('MockTokenizer', (), {
            'tokenize': lambda self, text: text.split()
        })()
        
        self.experiment.add_model("test_model", mock_model, mock_tokenizer)
        self.assertIn("test_model", self.experiment.models)
        
        self.experiment.remove_model("test_model")
        self.assertNotIn("test_model", self.experiment.models)
    
    def test_gate_experiment_without_models(self):
        """Test running gate experiment without models."""
        result = self.experiment.run_gate_experiment(
            "AND", "test text", True, False
        )
        
        self.assertEqual(result['gate'], "AND")
        self.assertEqual(result['logical_input'], True)
        self.assertEqual(result['emotional_input'], False)
        self.assertEqual(result['gate_result'], False)  # True AND False = False
        self.assertEqual(result['input_text'], "test text")
        self.assertIsInstance(result['model_results'], dict)
        self.assertEqual(len(result['model_results']), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_config_integration_with_classifier(self):
        """Test using config manager with token classifier."""
        # Create temporary config
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            config_manager.add_to_list("boolean_logic.logical_words", "integration_test")
            
            # Create classifier with config
            classifier = TokenClassifier(
                logical_words=config_manager.get("boolean_logic.logical_words")
            )
            
            # Test the integration
            self.assertEqual(classifier.classify_token("integration_test"), "logical")
    
    def test_full_experiment_pipeline(self):
        """Test complete experiment pipeline."""
        # Setup
        config_manager = ConfigManager()
        gate_lib = BooleanGateLibrary()
        classifier = TokenClassifier(
            logical_words=config_manager.get("boolean_logic.logical_words"),
            emotional_words=config_manager.get("boolean_logic.emotional_words")
        )
        experiment = BooleanLogicExperiment(gate_lib, classifier)
        
        # Run experiment
        result = experiment.run_gate_experiment(
            "XOR", "I love this logical analysis", True, True
        )
        
        # Verify results
        self.assertEqual(result['gate'], "XOR")
        self.assertEqual(result['gate_result'], False)  # True XOR True = False
        self.assertIn("input_text", result)
        self.assertEqual(len(experiment.results), 1)


def run_basic_functionality_test():
    """Run a quick smoke test of basic functionality."""
    print("Running basic functionality test...")
    
    try:
        # Test gate library
        gate_lib = BooleanGateLibrary()
        assert gate_lib.evaluate_gate("AND", True, True) == True
        print("✓ Boolean gate library working")
        
        # Test classifier
        classifier = TokenClassifier()
        assert classifier.classify_token("love") == "emotional"
        assert classifier.classify_token("therefore") == "logical"
        print("✓ Token classifier working")
        
        # Test config manager
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(temp_dir)
            config_manager.set("test.value", "success")
            assert config_manager.get("test.value") == "success"
        print("✓ Configuration manager working")
        
        print("✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


if __name__ == "__main__":
    # Run smoke test first
    if not run_basic_functionality_test():
        sys.exit(1)
    
    # Run full test suite
    print("\nRunning full test suite...")
    unittest.main(argv=[''], verbosity=2, exit=False)