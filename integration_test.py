#!/usr/bin/env python3
"""
Integration test demonstrating the unified AI Research Collection
Tests the interaction between all new components.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boolean_logic_lib import BooleanGateLibrary, TokenClassifier, BooleanLogicExperiment
from agent_system import create_literature_agent, create_boolean_logic_agent, AgentOrchestrator, AgentFaculty
from config_manager import ConfigManager

def test_boolean_logic_integration():
    """Test boolean logic with configuration."""
    print("=== Boolean Logic Integration Test ===")
    
    # Setup configuration
    config = ConfigManager()
    config.load_config()
    
    # Create components with config
    gate_lib = BooleanGateLibrary()
    classifier = TokenClassifier(
        logical_words=config.get('boolean_logic.logical_words'),
        emotional_words=config.get('boolean_logic.emotional_words')
    )
    
    # Test gate operations
    print(f"AND(True, False) = {gate_lib.evaluate_gate('AND', True, False)}")
    print(f"XOR(True, True) = {gate_lib.evaluate_gate('XOR', True, True)}")
    
    # Test classification
    test_text = "I love this logical analysis because the data shows clear evidence"
    tokens = test_text.split()
    classifications = classifier.classify_tokens_batch(tokens, test_text)
    
    print(f"\nText: {test_text}")
    print("Token classifications:")
    for token, classification in zip(tokens, classifications):
        print(f"  {token}: {classification}")
    
    # Test experiment (without models)
    experiment = BooleanLogicExperiment(gate_lib, classifier)
    result = experiment.run_gate_experiment("OR", test_text, True, True)
    print(f"\nExperiment result: {result['gate_result']}")
    
    print("✓ Boolean logic integration successful\n")

def test_agent_system_integration():
    """Test agent system with memory and faculties."""
    print("=== Agent System Integration Test ===")
    
    # Create agents
    lit_agent = create_literature_agent("TestLitAgent")
    bool_agent = create_boolean_logic_agent("TestBoolAgent")
    
    # Setup boolean agent with gate library
    from boolean_logic_lib import BooleanGateLibrary
    bool_agent.set_gate_library(BooleanGateLibrary())
    
    # Add some documents to literature agent
    lit_agent.add_document("sample.txt", "This is a test document about love and logical reasoning.")
    
    # Test agent interactions
    print("Testing Literature Agent:")
    action1 = lit_agent.process_input("recall information about literature")
    print(f"  {action1.faculty.value}: {action1.content}")
    
    action2 = lit_agent.process_input("consult sample.txt")
    print(f"  {action2.faculty.value}: {action2.content}")
    
    action3 = lit_agent.process_input("analyze the themes")
    print(f"  {action3.faculty.value}: {action3.content}")
    
    print("\nTesting Boolean Agent:")
    action4 = bool_agent.process_input("what gates are available?")
    print(f"  {action4.faculty.value}: {action4.content}")
    
    action5 = bool_agent.process_input("explain truth tables")
    print(f"  {action5.faculty.value}: {action5.content}")
    
    # Test orchestrator
    orchestrator = AgentOrchestrator()
    orchestrator.add_agent(lit_agent)
    orchestrator.add_agent(bool_agent)
    
    status = orchestrator.get_agent_status()
    print(f"\nAgent Status: {len(status)} agents registered")
    for name, stat in status.items():
        print(f"  {name}: {stat['action_count']} actions, {stat['memory_count']} memories")
    
    print("✓ Agent system integration successful\n")

def test_configuration_integration():
    """Test configuration management."""
    print("=== Configuration Integration Test ===")
    
    config = ConfigManager()
    
    # Test getting default values
    logical_words = config.get('boolean_logic.logical_words')
    print(f"Default logical words count: {len(logical_words)}")
    
    # Test setting values
    config.set('test.integration', 'success')
    value = config.get('test.integration')
    print(f"Set and retrieved value: {value}")
    
    # Test list operations
    original_count = len(config.get('boolean_logic.logical_words'))
    config.add_to_list('boolean_logic.logical_words', 'integration_test_word')
    new_count = len(config.get('boolean_logic.logical_words'))
    print(f"Added word to list: {original_count} -> {new_count}")
    
    print("✓ Configuration integration successful\n")

def test_full_workflow():
    """Test a complete workflow using all components."""
    print("=== Full Workflow Integration Test ===")
    
    # Setup
    config = ConfigManager()
    config.load_config()
    
    # Create boolean logic components
    gate_lib = BooleanGateLibrary()
    classifier = TokenClassifier(
        logical_words=config.get('boolean_logic.logical_words'),
        emotional_words=config.get('boolean_logic.emotional_words')
    )
    
    # Create an agent with boolean logic capability
    agent = create_boolean_logic_agent("WorkflowAgent")
    agent.set_gate_library(gate_lib)
    
    # Add some domain knowledge
    agent.add_memory("Boolean logic is fundamental to computer science", "knowledge", ["boolean", "computer_science"])
    agent.add_memory("XOR gates are useful for comparing inputs", "knowledge", ["gates", "comparison"])
    
    # Simulate a workflow
    print("Starting workflow simulation...")
    
    # Step 1: Agent recalls relevant information
    recall_action = agent.execute_faculty(AgentFaculty.RECALL, "boolean gates")
    print(f"Step 1 - Recall: {recall_action.content[:100]}...")
    
    # Step 2: Agent thinks about the problem
    think_action = agent.execute_faculty(AgentFaculty.THINK, "How can XOR gates be used in logic experiments?")
    print(f"Step 2 - Think: {think_action.content[:100]}...")
    
    # Step 3: Evaluate some gates
    test_cases = [
        ("AND", True, False),
        ("OR", False, True),
        ("XOR", True, True)
    ]
    
    for gate, input1, input2 in test_cases:
        result = gate_lib.evaluate_gate(gate, input1, input2)
        agent.add_memory(f"{gate} gate with inputs {input1}, {input2} produces {result}", "experiment", ["gates", gate.lower()])
    
    # Step 4: Agent provides final response
    talk_action = agent.execute_faculty(AgentFaculty.TALK, "I have completed the boolean logic analysis")
    print(f"Step 4 - Talk: {talk_action.content}")
    
    # Step 5: Classify some text using the classifier
    test_text = "The logical analysis shows that emotional responses can influence reasoning"
    tokens = test_text.split()
    classifications = classifier.classify_tokens_batch(tokens, test_text)
    
    logical_count = classifications.count('logical')
    emotional_count = classifications.count('emotional')
    neutral_count = classifications.count('neutral')
    
    print(f"Step 5 - Classification: {logical_count} logical, {emotional_count} emotional, {neutral_count} neutral tokens")
    
    # Final status
    print(f"Workflow complete - Agent performed {len(agent.action_history)} actions")
    print("✓ Full workflow integration successful\n")

def main():
    """Run all integration tests."""
    print("AI Research Collection - Integration Test Suite")
    print("=" * 50)
    
    try:
        test_boolean_logic_integration()
        test_agent_system_integration() 
        test_configuration_integration()
        test_full_workflow()
        
        print("🎉 All integration tests passed successfully!")
        print("\nThe AI Research Collection unified architecture is working correctly.")
        print("You can now use:")
        print("  - python cli.py (for interactive interface)")
        print("  - python test_ai_research.py (for unit tests)")
        print("  - Individual components as needed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)