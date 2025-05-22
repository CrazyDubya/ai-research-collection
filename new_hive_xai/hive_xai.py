import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
import random
import openai
import requests
import tiktoken
import os
from typing import Dict, List, Optional

# Constants
XAI_API_KEY = os.getenv("XAI_API_KEY", "your-xai-api-key-here")  # Set this in your environment
OLLAMA_URL = "http://localhost:11434/api/generate"
OPENAI_API_URL = "https://api.x.ai/v1"

# Abstract LLM Interface
class LLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int) -> str:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

# OpenAI LLM Implementation (for xAI's Grok-2)
class OpenAILLM(LLM):
    def __init__(self):
        self.client = openai.OpenAI(base_url=OPENAI_API_URL, api_key=XAI_API_KEY)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_response(self, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "You are Grok, a helpful AI for exploring math and reasoning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

# Ollama LLM Implementation (for long-gemma)
class OllamaLLM(LLM):
    def generate_response(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": "long-gemma",
            "prompt": prompt,
            "max_tokens": max_tokens
        }
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"].strip()
        return "Error: Unable to connect to ollama"

    def count_tokens(self, text: str) -> int:
        return len(text.split())  # Rough estimate

# Base Node (Agent)
class Node:
    def __init__(self, node_id: str, llm: LLM, total_tokens: int, tokens_per_cycle: int, cycles: int, task: str):
        self.node_id = node_id
        self.llm = llm
        self.total_tokens = total_tokens
        self.tokens_per_cycle = tokens_per_cycle
        self.cycles = cycles
        self.remaining_tokens = total_tokens
        self.current_cycle = 0
        self.task = task
        self.xml = self._init_xml()
        self.sub_agents: List[Node] = []
        self.history = []

    def _init_xml(self) -> ET.Element:
        root = ET.Element("agent", id=self.node_id)
        ET.SubElement(root, "task").text = self.task
        ET.SubElement(root, "guidelines").text = "Explore and solve step-by-step"
        resources = ET.SubElement(root, "resources")
        ET.SubElement(resources, "total_tokens").text = str(self.total_tokens)
        ET.SubElement(resources, "tokens_per_cycle").text = str(self.tokens_per_cycle)
        ET.SubElement(resources, "cycles").text = str(self.cycles)
        ET.SubElement(resources, "remaining_tokens").text = str(self.remaining_tokens)
        expectations = ET.SubElement(root, "expectations")
        ET.SubElement(expectations, "metric", name="accuracy", target="0.9")
        prompt_design = ET.SubElement(root, "prompt_design")
        ET.SubElement(prompt_design, "template").text = f"Given a {self.task}, solve it step-by-step and explain your reasoning"
        ET.SubElement(prompt_design, "exploration_strategy").text = "Try one new prompt variation per cycle, keep if accuracy improves"
        ET.SubElement(root, "sub_agents")
        ET.SubElement(root, "history")
        return root

    def save_xml(self, filename: str):
        tree = ET.ElementTree(self.xml)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    def load_xml(self, filename: str):
        tree = ET.parse(filename)
        self.xml = tree.getroot()
        self.task = self.xml.find("task").text
        resources = self.xml.find("resources")
        self.total_tokens = int(resources.find("total_tokens").text)
        self.tokens_per_cycle = int(resources.find("tokens_per_cycle").text)
        self.cycles = int(resources.find("cycles").text)
        self.remaining_tokens = int(resources.find("remaining_tokens").text)

    def start_new_cycle(self) -> bool:
        if self.current_cycle < self.cycles and self.remaining_tokens >= self.tokens_per_cycle:
            self.current_cycle += 1
            return True
        return False

    def can_perform_task(self) -> bool:
        return self.remaining_tokens >= self.tokens_per_cycle

    def process_task(self, input_task: str) -> str:
        if not self.can_perform_task():
            return "Insufficient tokens"
        prompt = self.xml.find("prompt_design/template").text.replace(self.task, input_task)
        tokens_used = self.llm.count_tokens(prompt)
        if tokens_used > self.tokens_per_cycle:
            return "Prompt exceeds token limit"
        response = self.llm.generate_response(prompt, self.tokens_per_cycle - tokens_used)
        total_tokens = tokens_used + self.llm.count_tokens(response)
        self.remaining_tokens -= total_tokens
        self.xml.find("resources/remaining_tokens").text = str(self.remaining_tokens)
        return response

    def explore(self, verbose: int = 1):
        if not self.can_perform_task():
            if verbose >= 1:
                print(f"[{self.node_id}] Cycle {self.current_cycle}: Cannot explore, insufficient tokens.")
            return

        # Generate a random math problem
        problem_type = random.choice(["linear", "quadratic"])
        if problem_type == "linear":
            a, b = random.randint(1, 10), random.randint(1, 10)
            solution = 3  # Fixed for simplicity
            problem = f"{a}x + {b} = {a * solution + b}"
            expected = f"x = {solution}"
        else:
            a = random.randint(1, 5)
            problem = f"x^2 = {a * a}"
            expected = f"x = {a} or x = -{a}"

        # Process the task
        if verbose >= 2 or (verbose == 1 and self.current_cycle % 3 == 0):
            print(f"[{self.node_id}] Cycle {self.current_cycle}: Solving '{problem}'")
        response = self.process_task(problem)
        accuracy = 1.0 if expected in response else 0.0  # Simplified evaluation
        self.history.append({"task": problem, "response": response, "accuracy": accuracy})
        self.update_xml_history(problem, accuracy)

        if verbose >= 2 or (verbose == 1 and self.current_cycle % 3 == 0):
            print(f"[{self.node_id}] Response: '{response}' (Accuracy: {accuracy})")

        # Explore prompt variation (30% chance)
        if random.random() < 0.3 and self.can_perform_task():
            new_prompt = f"Solve {problem} with clear examples and steps"
            if verbose >= 2 or (verbose == 1 and self.current_cycle % 3 == 0):
                print(f"[{self.node_id}] Exploring new prompt: '{new_prompt}'")
            response = self.llm.generate_response(new_prompt, self.tokens_per_cycle)
            self.remaining_tokens -= self.llm.count_tokens(new_prompt) + self.llm.count_tokens(response)
            if verbose >= 2 or (verbose == 1 and self.current_cycle % 3 == 0):
                print(f"[{self.node_id}] New prompt response: '{response}'")
            if "x =" in response:
                self.xml.find("prompt_design/template").text = f"Given a {self.task}, solve with clear examples and steps"
                if verbose >= 1:
                    print(f"[{self.node_id}] Updated prompt template to: '{self.xml.find('prompt_design/template').text}'")

    def update_xml_history(self, task: str, accuracy: float):
        history = self.xml.find("history")
        entry = ET.SubElement(history, "task", id=f"T{len(self.history)}")
        ET.SubElement(entry, "outcome").text = "success" if accuracy >= 0.9 else "failure"
        ET.SubElement(entry, "accuracy").text = str(accuracy)

    def spawn_sub_agent(self, sub_task: str, token_fraction: float = 0.2, verbose: int = 1):
        sub_tokens = int(self.total_tokens * token_fraction)
        if sub_tokens > self.remaining_tokens:
            if verbose >= 1:
                print(f"[{self.node_id}] Cannot spawn sub-agent: insufficient tokens.")
            return None
        sub_node = Node(
            f"{self.node_id}_sub{random.randint(1, 1000)}",
            self.llm,
            sub_tokens,
            self.tokens_per_cycle,
            self.cycles // 2,
            sub_task
        )
        self.sub_agents.append(sub_node)
        self.remaining_tokens -= sub_tokens
        self.xml.find("resources/remaining_tokens").text = str(self.remaining_tokens)
        sub_agents = self.xml.find("sub_agents")
        ET.SubElement(sub_agents, "sub_agent", id=sub_node.node_id, allocated_tokens=str(sub_tokens))
        if verbose >= 1:
            print(f"[{self.node_id}] Spawned sub-agent {sub_node.node_id} for '{sub_task}' with {sub_tokens} tokens.")
        return sub_node

# Index Node
class IndexNode:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def load_node(self, node_id: str, filename: str):
        if node_id not in self.nodes:
            node = Node(node_id, OpenAILLM(), 0, 0, 0, "")  # Dummy init
            node.load_xml(filename)
            self.nodes[node_id] = node
        return self.nodes[node_id]

    def find_node_for_task(self, task: str) -> Optional[Node]:
        for node in self.nodes.values():
            if task in node.task:
                return node
        return None

# Human Node
class HumanNode:
    def __init__(self, index_node: IndexNode):
        self.index_node = index_node

    def provide_guidance(self, node_id: str, guidance: str):
        node = self.index_node.nodes.get(node_id)
        if node:
            node.xml.find("guidelines").text = guidance
            print(f"[{node_id}] Updated guidelines: {guidance}")

    def evaluate(self, node_id: str) -> Dict:
        node = self.index_node.nodes.get(node_id)
        if not node:
            return {"error": "Node not found"}
        total_accuracy = sum(h["accuracy"] for h in node.history) / len(node.history) if node.history else 0
        return {
            "node_id": node_id,
            "total_tasks": len(node.history),
            "average_accuracy": total_accuracy,
            "remaining_tokens": node.remaining_tokens,
            "cycles_used": node.current_cycle
        }

    def run_additional_cycles(self, node_id: str, cycles: int, guidance: str = None, verbose: int = 1):
        node = self.index_node.nodes.get(node_id)
        if node:
            if guidance:
                self.provide_guidance(node_id, guidance)
            for i in range(cycles):
                if node.start_new_cycle():
                    if verbose >= 1:
                        print(f"[{node_id}] Additional cycle {i + 1}/{cycles}")
                    node.explore(verbose=verbose)

# Self-Play Module (Fixed)
def self_play(index_node: IndexNode, cycles_per_node: int, verbose: int = 1):
    # Collect initial nodes to avoid dictionary size change during iteration
    initial_nodes = list(index_node.nodes.values())
    for node in initial_nodes:
        if verbose >= 1:
            print(f"[{node.node_id}] Starting self-play for {cycles_per_node} cycles...")
        for i in range(cycles_per_node):
            if node.start_new_cycle():
                if verbose >= 2 or (verbose == 1 and i % 3 == 0):
                    print(f"[{node.node_id}] Cycle {node.current_cycle}/{cycles_per_node}")
                node.explore(verbose=verbose)
                if random.random() < 0.1:  # 10% chance to spawn sub-agent
                    sub_node = node.spawn_sub_agent("Solve quadratic equations", verbose=verbose)
                    if sub_node:
                        index_node.add_node(sub_node)
                        # Sub-agent gets one exploration cycle immediately
                        if verbose >= 1:
                            print(f"[{sub_node.node_id}] Sub-agent starting exploration...")
                        sub_node.explore(verbose=verbose)
        if verbose >= 1:
            print(f"[{node.node_id}] Self-play completed.")

# Main Execution
if __name__ == "__main__":
    # Human-set budget
    total_tokens = 1_000_000  # 1 million tokens
    tokens_per_cycle = 500
    total_cycles = 1_000  # 1 thousand cycles

    # Initialize Index Node
    index = IndexNode()

    # Choose LLM provider (default to OpenAI/xAI Grok-2)
    llm_choice = "openai"  # Switch to "ollama" for long-gemma
    llm = OpenAILLM() if llm_choice == "openai" else OllamaLLM()

    # Create initial agent
    math_node = Node("Math001", llm, total_tokens, tokens_per_cycle, total_cycles, "math problem")
    index.add_node(math_node)
    math_node.save_xml("math_node.xml")
    print(f"Initialized {math_node.node_id} with {total_tokens} tokens and {total_cycles} cycles.")

    # Run self-play with verbose output
    print("Starting self-play...")
    self_play(index, 10, verbose=1)  # Snippets mode

    # Human interaction
    human = HumanNode(index)
    evaluation = human.evaluate("Math001")
    print("Evaluation:", evaluation)

    # Human decision loop
    while True:
        action = input("Accept (yes) or run more cycles (no)? ").lower()
        if action == "yes":
            break
        elif action == "no":
            guidance = input("Provide guidance (or press Enter for none): ")
            cycles = int(input("How many additional cycles? "))
            verbose = int(input("Verbose level (0 = none, 1 = snippets, 2 = full)? "))
            human.run_additional_cycles("Math001", cycles, guidance or None, verbose=verbose)
            evaluation = human.evaluate("Math001")
            print("Post-guidance evaluation:", evaluation)
        else:
            print("Invalid input. Try 'yes' or 'no'.")

    # Save final state
    math_node.save_xml("math_node_final.xml")
    print("System state saved to math_node_final.xml.")