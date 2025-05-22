import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
import random
import openai
import requests
import tiktoken
import os
from typing import Dict, List, Optional

# Constants
XAI_API_KEY = os.getenv("XAI_API_KEY", "your-xai-api-key-here")
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

class OpenAILLM(LLM):
    def __init__(self):
        self.client = openai.OpenAI(base_url=OPENAI_API_URL, api_key=XAI_API_KEY)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_response(self, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "You are Grok, a versatile AI with a dynamic personality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.9
        )
        return response.choices[0].message.content.strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

class OllamaLLM(LLM):
    def generate_response(self, prompt: str, max_tokens: int) -> str:
        payload = {"model": "long-gemma", "prompt": prompt, "max_tokens": max_tokens}
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"].strip()
        return "Error: Unable to connect to ollama"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

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
        self.personality = random.choice(["curious scientist", "witty philosopher", "playful storyteller"])
        self.xml = self._init_xml()
        self.sub_agents: List[Node] = []
        self.history = []
        self.task_evolution = [task]

    def _init_xml(self) -> ET.Element:
        root = ET.Element("agent", id=self.node_id)
        ET.SubElement(root, "task").text = self.task
        ET.SubElement(root, "guidelines").text = "Explore diverse reasoning and adapt creatively"
        ET.SubElement(root, "personality").text = self.personality
        resources = ET.SubElement(root, "resources")
        ET.SubElement(resources, "total_tokens").text = str(self.total_tokens)
        ET.SubElement(resources, "tokens_per_cycle").text = str(self.tokens_per_cycle)
        ET.SubElement(resources, "cycles").text = str(self.cycles)
        ET.SubElement(resources, "remaining_tokens").text = str(self.remaining_tokens)
        expectations = ET.SubElement(root, "expectations")
        ET.SubElement(expectations, "metric", name="engagement", target="0.8")
        prompt_design = ET.SubElement(root, "prompt_design")
        ET.SubElement(prompt_design, "template").text = f"As a {self.personality}, given a {self.task}, explore with a creative twist and detailed reasoning"
        ET.SubElement(prompt_design, "exploration_strategy").text = "Refine prompts based on engagement and insight"
        ET.SubElement(root, "sub_agents")
        ET.SubElement(root, "history")
        ET.SubElement(root, "task_evolution").text = self.task
        return root

    def save_xml(self, filename: str):
        tree = ET.ElementTree(self.xml)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    def load_xml(self, filename: str):
        tree = ET.parse(filename)
        self.xml = tree.getroot()
        self.task = self.xml.find("task").text
        self.personality = self.xml.find("personality").text
        resources = self.xml.find("resources")
        self.total_tokens = int(resources.find("total_tokens").text)
        self.tokens_per_cycle = int(resources.find("tokens_per_cycle").text)
        self.cycles = int(resources.find("cycles").text)
        self.remaining_tokens = int(resources.find("remaining_tokens").text)
        self.task_evolution = self.xml.find("task_evolution").text.split(", ")

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

        # Expanded task types with more sophisticated exploration paths
        task_types = [
            ("supervisory_analysis", self._generate_supervisory_task),
            ("primary_node_expansion", self._generate_primary_task),
            ("cross_node_synthesis", self._generate_synthesis_task),
            ("math problem", self._generate_math_problem),
            ("logic puzzle", self._generate_logic_puzzle),
            ("philosophical query", self._generate_philosophical_query),
            ("self-reflection", self._generate_self_reflection)
        ]

        # Weight towards supervisory and primary node tasks
        weights = [0.25, 0.25, 0.2, 0.075, 0.075, 0.075, 0.075]
        task_type, generator = random.choices(task_types, weights=weights)[0]
        problem, expected = generator()

        if verbose >= 2 or (verbose == 1 and self.current_cycle % 3 == 0):
            print(f"[{self.node_id}] Cycle {self.current_cycle}: Exploring '{task_type}': '{problem}'")
        
        response = self.process_task(problem)
        # More nuanced engagement scoring
        engagement = self._calculate_engagement(response, task_type)
        self.history.append({"task": problem, "response": response, "engagement": engagement, "type": task_type})
        self.update_xml_history(problem, engagement)

        if verbose >= 2 or (verbose == 1 and self.current_cycle % 3 == 0):
            print(f"[{self.node_id}] Response: '{response}' (Engagement: {engagement})")

        # More aggressive task evolution for supervisory and primary nodes
        evolution_chance = 0.4 if task_type in ["supervisory_analysis", "primary_node_expansion"] else 0.2
        if random.random() < evolution_chance and engagement > 0.7:
            new_task = self._evolve_task(task_type)
            if new_task not in self.task_evolution:
                self.task_evolution.append(new_task)
                self.task = ", ".join(self.task_evolution)
                self.xml.find("task").text = self.task
                self.xml.find("task_evolution").text = self.task
                if verbose >= 1:
                    print(f"[{self.node_id}] Evolved task scope to: '{self.task}'")

        # Dynamic sub-agent creation based on task performance
        if task_type in ["supervisory_analysis", "primary_node_expansion"] and engagement > 0.8:
            self._consider_creating_sub_agent(problem, response)

        # Enhanced prompt refinement
        if random.random() < 0.4 and self.can_perform_task():
            critique_prompt = self._generate_critique_prompt(problem, response, task_type)
            if verbose >= 2:
                print(f"[{self.node_id}] Critiquing: '{critique_prompt}'")
            critique = self.llm.generate_response(critique_prompt, self.tokens_per_cycle)
            self.remaining_tokens -= self.llm.count_tokens(critique_prompt) + self.llm.count_tokens(critique)
            self._update_prompt_template(critique)

    def _generate_math_problem(self) -> tuple:
        if random.choice([True, False]):
            a, b = random.randint(1, 10), random.randint(1, 10)
            solution = random.randint(1, 5)
            return f"{a}x + {b} = {a * solution + b}", f"x = {solution}"
        a = random.randint(1, 5)
        return f"x^2 = {a * a} - what's wrong with this and how can we fix it?", f"Clarify equation intent"

    def _generate_logic_puzzle(self) -> tuple:
        return "If A is true, then B is false. If B is true, then A is false. What can you conclude?", "A and B are mutually exclusive"

    def _generate_philosophical_query(self) -> tuple:
        return random.choice([
            ("What is the meaning of life in a universe that might be infinite?", "No fixed answer"),
            ("Can chaos and order coexist in a single system?", "Reasoned balance expected")
        ])

    def _generate_self_reflection(self) -> tuple:
        last_task = self.history[-1]["task"] if self.history else "my existence"
        return f"What can I learn from '{last_task}' to grow wiser?", "Insightful growth expected"

    def _generate_supervisory_task(self) -> tuple:
        tasks = [
            ("Analyze the patterns of successful node interactions and propose optimization strategies", "Strategic analysis"),
            ("Evaluate the current exploration paths and suggest new directions for investigation", "Path evaluation"),
            ("Synthesize insights from multiple sub-agents to form higher-level understanding", "Multi-agent synthesis")
        ]
        return random.choice(tasks)

    def _generate_primary_task(self) -> tuple:
        tasks = [
            ("Explore novel problem-solving approaches in the current domain", "Innovation"),
            ("Investigate potential connections between different knowledge domains", "Cross-domain linking"),
            ("Develop and test new hypotheses based on observed patterns", "Hypothesis generation")
        ]
        return random.choice(tasks)

    def _generate_synthesis_task(self) -> tuple:
        recent_tasks = [h["task"] for h in self.history[-5:]] if self.history else ["base knowledge"]
        return f"Synthesize insights from recent explorations: {', '.join(recent_tasks[:2])}", "Synthesis"

    def _calculate_engagement(self, response: str, task_type: str) -> float:
        """Calculate engagement score based on response quality and task type."""
        try:
            if not response:
                return 0.0

            # Calculate base metrics
            words = response.split()
            if not words:
                return 0.0

            # Normalize response length (cap at 500 words)
            base_score = min(len(words) / 200, 2.5)  # Allows for scores up to 1.0 after multiplier
            
            # Calculate lexical diversity (unique words / total words)
            unique_words = len(set(words))
            complexity_score = unique_words / len(words)
            
            # Task type multiplier
            task_multiplier = 1.2 if task_type in ["supervisory_analysis", "primary_node_expansion"] else 1.0
            
            # Additional factors
            depth_indicators = ["because", "therefore", "however", "although", "furthermore"]
            depth_score = sum(1 for word in words if word.lower() in depth_indicators) / len(words)
            
            # Weighted combination
            raw_score = (
                base_score * 0.4 +
                complexity_score * 0.3 +
                depth_score * 0.3
            ) * task_multiplier
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, raw_score))
            
        except Exception as e:
            print(f"Warning: Error calculating engagement score: {e}")
            return 0.0

    def _consider_creating_sub_agent(self, problem: str, response: str) -> None:
        if len(self.sub_agents) < 5 and self.remaining_tokens > self.total_tokens * 0.3:
            sub_agent_id = f"{self.node_id}_sub_{len(self.sub_agents)}"
            sub_agent = Node(sub_agent_id, self.llm, 
                           total_tokens=self.total_tokens // 4,
                           tokens_per_cycle=self.tokens_per_cycle,
                           cycles=self.cycles // 2,
                           task=f"Specialized exploration derived from: {problem[:50]}...")
            self.sub_agents.append(sub_agent)

    def _generate_critique_prompt(self, problem: str, response: str, task_type: str) -> str:
        prompts = [
            f"Analyze the depth and innovation in the response to '{problem}'. How can we push the boundaries further?",
            f"What unexplored angles or connections were missed in addressing '{problem}'?",
            f"How could the response to '{problem}' be more impactful for {task_type}?"
        ]
        return random.choice(prompts)

    def _update_prompt_template(self, critique: str) -> None:
        keywords = ["innovative", "creative", "detailed", "systematic", "analytical"]
        if any(keyword in critique.lower() for keyword in keywords):
            new_template = f"As a {self.personality} focused on {', '.join(self.task_evolution[-2:] if len(self.task_evolution) > 1 else [self.task])}, "
            new_template += "explore with unprecedented depth and systematic innovation, connecting insights across domains"
            self.xml.find("prompt_design/template").text = new_template

    def _evolve_task(self, task_type: str) -> str:
        try:
            if not self.history:
                return f"initial_{task_type}_exploration"

            last_task = self.history[-1].get("task", "")

            if task_type == "supervisory_analysis":
                return f"meta_analysis_level_{len(self.task_evolution) + 1}"
            elif task_type == "primary_node_expansion":
                return f"primary_exploration_branch_{len(self.task_evolution) + 1}"
            elif task_type == "cross_node_synthesis":
                return f"synthesis_pathway_{len(self.task_evolution) + 1}"
            elif task_type == "math problem" and "x^2" in last_task:
                return "nonlinear equation mysteries"
            elif task_type == "philosophical query" and "infinite" in last_task:
                return "philosophy of infinity"
            elif task_type == "logic puzzle":
                return "advanced logical reasoning"
            else:
                return f"evolved_{task_type}_path_{len(self.task_evolution) + 1}"
        except Exception as e:
            print(f"Warning: Error evolving task for node {self.node_id}: {e}")
            return f"basic_{task_type}_exploration"
        return f"{task_type} deepening"

    def update_xml_history(self, task: str, engagement: float):
        """Update the XML history with task results and engagement metrics."""
        try:
            history = self.xml.find("history")
            if history is None:
                history = ET.SubElement(self.xml, "history")
            
            entry = ET.SubElement(history, "task", id=f"T{len(self.history)}")
            ET.SubElement(entry, "content").text = task
            ET.SubElement(entry, "outcome").text = "engaging" if engagement >= 0.8 else "dull"
            ET.SubElement(entry, "engagement").text = str(engagement)
            
            # Update task evolution in XML
            task_evolution = self.xml.find("task_evolution")
            if task_evolution is not None:
                task_evolution.text = ", ".join(self.task_evolution)
        except Exception as e:
            print(f"Warning: Error updating XML history for node {self.node_id}: {e}")

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

class IndexNode:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.supervisors: Dict[str, List[str]] = {}  # supervisor_id -> list of supervised node_ids
        self.branches: Dict[str, List[str]] = {}  # branch_name -> list of node_ids
        self.node_metrics: Dict[str, Dict] = {}  # node_id -> performance metrics
        self.exploration_paths: Dict[str, List[str]] = {}  # path_id -> sequence of node_ids

    def add_node(self, node: Node, supervisor_id: Optional[str] = None, branch: Optional[str] = None):
        self.nodes[node.node_id] = node
        
        # Update supervisor relationships
        if supervisor_id:
            if supervisor_id not in self.supervisors:
                self.supervisors[supervisor_id] = []
            self.supervisors[supervisor_id].append(node.node_id)
        
        # Update branch information
        if branch:
            if branch not in self.branches:
                self.branches[branch] = []
            self.branches[branch].append(node.node_id)
        
        # Initialize metrics
        self.node_metrics[node.node_id] = {
            "avg_engagement": 0.0,
            "successful_cycles": 0,
            "total_cycles": 0,
            "branching_factor": 0,
            "innovation_score": 0.0
        }

    def create_exploration_path(self, path_id: str, root_node_id: str):
        self.exploration_paths[path_id] = [root_node_id]

    def extend_exploration_path(self, path_id: str, node_id: str):
        if path_id in self.exploration_paths:
            self.exploration_paths[path_id].append(node_id)

    def get_branch_performance(self, branch: str) -> float:
        if branch not in self.branches or not self.branches[branch]:
            return 0.0
        return sum(self.node_metrics[node_id]["avg_engagement"] 
                  for node_id in self.branches[branch]) / len(self.branches[branch])

    def get_supervisor_effectiveness(self, supervisor_id: str) -> float:
        if supervisor_id not in self.supervisors or not self.supervisors[supervisor_id]:
            return 0.0
        supervised_nodes = self.supervisors[supervisor_id]
        return sum(self.node_metrics[node_id]["avg_engagement"] 
                  for node_id in supervised_nodes) / len(supervised_nodes)

    def update_node_metrics(self, node_id: str, metrics: Dict):
        if node_id in self.node_metrics:
            self.node_metrics[node_id].update(metrics)

    def load_node(self, node_id: str, filename: str):
        if node_id not in self.nodes:
            node = Node(node_id, OpenAILLM(), 0, 0, 0, "")
            node.load_xml(filename)
            self.nodes[node_id] = node
        return self.nodes[node_id]

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
        total_engagement = sum(h["engagement"] for h in node.history) / len(node.history) if node.history else 0
        return {
            "node_id": node_id,
            "total_tasks": len(node.history),
            "average_engagement": total_engagement,
            "current_task_scope": node.task,
            "personality": node.personality,
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
                    if i % 10 == 0 and verbose >= 1:  # Summary every 10 cycles
                        print(f"[{node_id}] Summary: Task scope='{node.task}', Engagement={sum(h['engagement'] for h in node.history[-10:]) / 10:.2f}")

def self_play(index_node: IndexNode, cycles_per_node: int, verbose: int = 1, min_nodes: int = 10):
    """Run extended self-play with dynamic node creation and exploration."""
    # Get an existing LLM or create a new one
    existing_llm = None
    for node in index_node.nodes.values():
        if node.llm is not None:
            existing_llm = node.llm
            break
    
    if existing_llm is None:
        existing_llm = OpenAILLM()
    
    # Create initial supervisor nodes
    num_supervisors = max(2, min_nodes // 5)
    for i in range(num_supervisors):
        supervisor_id = f"supervisor_{i}"
        supervisor_node = Node(supervisor_id, existing_llm,
                             total_tokens=1_500_000,  # More tokens for supervisors
                             tokens_per_cycle=750,
                             cycles=cycles_per_node,
                             task=f"high_level_supervision_branch_{i}")
        index_node.add_node(supervisor_node)
        index_node.create_exploration_path(f"path_{i}", supervisor_id)
    
    # Create initial branch nodes under supervisors
    branches = ["reasoning", "creativity", "analysis", "synthesis", "innovation"]
    for branch in branches:
        supervisor_id = random.choice([n for n in index_node.nodes.keys() if n.startswith("supervisor")])
        branch_node = Node(f"branch_{branch}", existing_llm,
                          total_tokens=1_000_000,
                          tokens_per_cycle=500,
                          cycles=cycles_per_node,
                          task=f"specialized_{branch}_exploration")
        index_node.add_node(branch_node, supervisor_id=supervisor_id, branch=branch)
    
    # Ensure minimum number of exploration nodes
    while len(index_node.nodes) < min_nodes:
        # Choose a branch and supervisor for the new node
        branch = random.choice(branches)
        supervisor_id = random.choice([n for n in index_node.nodes.keys() if n.startswith("supervisor")])
        
        node_id = f"node_{len(index_node.nodes)}"
        task_type = random.choice(["primary", "support"])
        base_task = f"{task_type}_exploration_in_{branch}_{len(index_node.nodes)}"
        
        new_node = Node(node_id, existing_llm,
                       total_tokens=1_000_000,
                       tokens_per_cycle=500,
                       cycles=cycles_per_node,
                       task=base_task)
        
        index_node.add_node(new_node, supervisor_id=supervisor_id, branch=branch)
        # Add to an exploration path
        path_id = f"path_{len(index_node.nodes) % num_supervisors}"
        index_node.extend_exploration_path(path_id, node_id)

    # Track performance and relationships
    node_performance = {}
    node_relationships = {}

    # Extended self-play with multiple phases
    for phase in range(3):  # Multiple phases for deeper exploration
        print(f"\nStarting self-play phase {phase + 1}")
        
        # First, run supervisor nodes to guide exploration
        supervisor_insights = {}
        for supervisor_id in [n for n in index_node.nodes.keys() if n.startswith("supervisor")]:
            supervisor = index_node.nodes[supervisor_id]
            print(f"\nSupervisor {supervisor_id} analyzing network...")
            
            # Analyze supervised nodes
            if supervisor_id in index_node.supervisors:
                supervised_nodes = index_node.supervisors[supervisor_id]
                for node_id in supervised_nodes:
                    node = index_node.nodes[node_id]
                    if node.history:
                        avg_engagement = sum(h.get("engagement", 0) for h in node.history[-5:]) / min(5, len(node.history))
                        supervisor_insights[node_id] = {
                            "engagement": avg_engagement,
                            "innovation_needed": avg_engagement < 0.7,
                            "branch_extension": avg_engagement > 0.8
                        }
        
        # Explore nodes with supervisor guidance
        for branch in index_node.branches:
            print(f"\nExploring {branch} branch")
            branch_nodes = index_node.branches[branch]
            
            # Sort nodes by potential (less explored nodes first)
            branch_nodes.sort(key=lambda n: len(index_node.nodes[n].history))
            
            for node_id in branch_nodes:
                node = index_node.nodes[node_id]
                print(f"\nExploring node {node_id} in {branch} branch")
                cycle_count = 0
                successful_cycles = 0
                
                while cycle_count < cycles_per_node:
                    if node.start_new_cycle():
                        node.explore(verbose=verbose)
                        cycle_count += 1
                        successful_cycles += 1
                        
                        # Consider creating new nodes based on supervisor insights
                        if successful_cycles % 5 == 0 and node.history:  # Check more frequently
                            try:
                                recent_engagement = sum(h.get("engagement", 0) for h in node.history[-5:]) / min(5, len(node.history))
                                
                                # Update node metrics
                                index_node.update_node_metrics(node_id, {
                                    "avg_engagement": recent_engagement,
                                    "successful_cycles": successful_cycles,
                                    "total_cycles": cycle_count
                                })
                                
                                # Create new nodes based on performance and supervisor insights
                                if recent_engagement > 0.75:
                                    # Choose creation strategy based on branch performance
                                    branch_perf = index_node.get_branch_performance(branch)
                                    if branch_perf > 0.7:
                                        # Branch is performing well, create specialized node
                                        new_node_id = f"specialized_{branch}_{len(index_node.nodes)}"
                                        new_task = f"deep_{branch}_exploration_phase_{phase}"
                                    else:
                                        # Branch needs diversity, create hybrid node
                                        other_branch = random.choice([b for b in index_node.branches if b != branch])
                                        new_node_id = f"hybrid_{branch}_{other_branch}_{len(index_node.nodes)}"
                                        new_task = f"cross_{branch}_{other_branch}_exploration"
                                    
                                    supervisor_id = random.choice([n for n in index_node.nodes.keys() if n.startswith("supervisor")])
                                    new_node = Node(new_node_id, node.llm, 
                                                   total_tokens=1_000_000,
                                                   tokens_per_cycle=500,
                                                   cycles=cycles_per_node,
                                                   task=new_task)
                                    
                                    index_node.add_node(new_node, supervisor_id=supervisor_id, branch=branch)
                                    node_relationships[new_node_id] = node_id
                                    
                                    # Add to exploration path
                                    for path_id, path in index_node.exploration_paths.items():
                                        if node_id in path:
                                            index_node.extend_exploration_path(path_id, new_node_id)
                                            break
                                    
                                    print(f"Created new node {new_node_id} in {branch} branch")
                            
                            except Exception as e:
                                print(f"Warning: Error in node creation for {node_id}: {e}")
                    else:
                        print(f"Node {node_id} cannot start new cycle")
                        break
                
                # Record node performance
                node_performance[node_id] = successful_cycles / cycle_count if cycle_count > 0 else 0
        
        # Phase summary with detailed metrics
        print(f"\nPhase {phase + 1} Summary:")
        print(f"Active Nodes: {len(index_node.nodes)}")
        print(f"Average Node Performance: {sum(node_performance.values()) / len(node_performance):.2f}")
        
        # Branch performance summary
        print("\nBranch Performance:")
        for branch in index_node.branches:
            perf = index_node.get_branch_performance(branch)
            print(f"{branch}: {perf:.2f}")
        
        # Supervisor effectiveness
        print("\nSupervisor Effectiveness:")
        for supervisor_id in index_node.supervisors:
            effectiveness = index_node.get_supervisor_effectiveness(supervisor_id)
            print(f"{supervisor_id}: {effectiveness:.2f}")
    
    if verbose >= 1:
        print("Self-play completed.")
    return node_performance, node_relationships

if __name__ == "__main__":
    total_tokens = 1_000_000
    tokens_per_cycle = 500
    total_cycles = 1_000

    index = IndexNode()
    llm_choice = "openai"  # Switch to "ollama"
    llm = OpenAILLM() if llm_choice == "openai" else OllamaLLM()

    explorer_node = Node("Explorer001", llm, total_tokens, tokens_per_cycle, total_cycles, "reasoning challenge")
    index.add_node(explorer_node)
    explorer_node.save_xml("explorer_node.xml")
    print(f"Initialized {explorer_node.node_id} with {total_tokens} tokens and {total_cycles} cycles.")

    print("Starting self-play...")
    self_play(index, 10, verbose=1)

    human = HumanNode(index)
    evaluation = human.evaluate("Explorer001")
    print("Evaluation:", evaluation)

    while True:
        action = input("Accept (yes) or run more cycles (no)? ").lower()
        if action == "yes":
            break
        elif action == "no":
            guidance = input("Provide guidance (or press Enter for none): ")
            cycles = int(input("How many additional cycles? "))
            verbose = int(input("Verbose level (0 = none, 1 = snippets, 2 = full)? "))
            human.run_additional_cycles("Explorer001", cycles, guidance or None, verbose=verbose)
            evaluation = human.evaluate("Explorer001")
            print("Post-guidance evaluation:", evaluation)
        else:
            print("Invalid input. Try 'yes' or 'no'.")

    explorer_node.save_xml("explorer_node_final.xml")
    print("System state saved to explorer_node_final.xml.")