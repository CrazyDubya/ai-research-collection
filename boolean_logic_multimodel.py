import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import pytest

# --- Pure Boolean Gate Definitions & Truth Table Generator ---
PURE_GATES = {
    "AND": (2, lambda a, b: a and b),
    "OR": (2, lambda a, b: a or b),
    "NAND": (2, lambda a, b: not (a and b)),
    "NOR": (2, lambda a, b: not (a or b)),
    "XOR": (2, lambda a, b: (a and not b) or (not a and b)),
    "XNOR": (2, lambda a, b: (a and b) or (not a and not b)),
    "NOT": (1, lambda a: not a)
}

def generate_truth_tables():
    """Generate and print Markdown truth tables for all pure boolean gates."""
    for name, (arity, func) in PURE_GATES.items():
        if arity == 2:
            print(f"### {name} Gate Truth Table\n| A | B | {name} |\n|:-:|:-:|:-:|")
            for A in (0, 1):
                for B in (0, 1):
                    result = int(func(A, B))
                    print(f"| {A} | {B} | {result} |")
            print()
        else:
            print(f"### {name} Gate Truth Table\n| A | {name} |\n|:-:|:-:|")
            for A in (0, 1):
                result = int(func(A))
                print(f"| {A} | {result} |")
            print()

# Prevent HuggingFace tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("boolean_logic.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "distilgpt2": {
        "name": "distilgpt2",
        "description": "Small GPT-2 model (82M parameters)",
        "huggingface_id": "distilgpt2",
        "tokenizer_id": "distilgpt2",
        "precision": "fp32",
        "size_mb": 330,
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "description": "GPT-2 Large (774M parameters)",
        "huggingface_id": "gpt2-large",
        "tokenizer_id": "gpt2-large",
        "precision": "fp32",
        "size_mb": 3000,
    },
    "llama2-7b": {
        "name": "llama2-7b",
        "description": "LLaMA 2 7B model",
        "huggingface_id": "meta-llama/Llama-2-7b-hf",
        "tokenizer_id": "meta-llama/Llama-2-7b-hf",
        "precision": "bf16",
        "size_mb": 13000,
    },
    "mistral-7b": {
        "name": "mistral-7b",
        "description": "Mistral 7B model",
        "huggingface_id": "mistralai/Mistral-7B-v0.1",
        "tokenizer_id": "mistralai/Mistral-7B-v0.1",
        "precision": "bf16",
        "size_mb": 14000,
    },
    "pythia-6.9b": {
        "name": "pythia-6.9b",
        "description": "EleutherAI Pythia 6.9B model",
        "huggingface_id": "EleutherAI/pythia-6.9b",
        "tokenizer_id": "EleutherAI/pythia-6.9b",
        "precision": "bf16",
        "size_mb": 13500,
    }
}

# Expanded logical words list
logical_words = [
    # Original terms
    "if", "then", "because", "therefore", "thus", "prove", "reason", "fact",
    "analyze", "conclude", "evidence", "logic", "deduction", "objective",

    # Academic/Scientific terms
    "research", "study", "data", "analysis", "method", "result", "conclusion",
    "hypothesis", "theory", "experiment", "observation", "empirical", "statistical",
    "measure", "calculate", "equation", "variable", "constant", "function", "correlation",
    "causation", "significant", "demonstrate", "indicate", "suggest", "implies",
    "consequently", "furthermore", "moreover", "additionally", "alternatively",

    # Reasoning terms
    "argument", "premise", "inference", "deductive", "inductive", "syllogism",
    "proposition", "assumption", "postulate", "axiom", "corollary", "theorem",
    "evaluate", "assess", "verify", "validate", "falsify", "refute", "disprove",
    "rational", "logical", "coherent", "consistent", "systematic", "methodical",

    # Common technical terms from samples
    "mechanism", "calculated", "measured", "discovered", "conventional",
    "analyzed", "developed", "identified", "investigated", "determined",
    "reactor", "genes", "mitochondrial", "carbon", "chemical", "antibodies",
    "neural", "clinical", "tests", "benzene",

    # Comparative/analytical words
    "compare", "contrast", "differentiate", "distinguish", "categorize",
    "classify", "organize", "structure", "framework", "model", "paradigm",
    "relative", "absolute", "specific", "general", "particular", "universal",

    # Scientific/technical terms
    "gene", "protein", "molecule", "vaccine", "virus", "cell",
    "system", "technology", "algorithm", "component"
]

# Expanded emotional words list
emotional_words = [
    # Original terms
    "love", "hate", "happy", "sad", "fear", "hope", "dream", "feel",
    "angry", "excited", "nervous", "proud", "passion", "joy",

    # Basic emotions
    "afraid", "anxious", "content", "depressed", "disappointed", "disgusted",
    "ecstatic", "embarrassed", "enraged", "envious", "frustrated", "grateful",
    "grief", "guilt", "heartbroken", "jealous", "joyful", "lonely", "nostalgic",
    "optimistic", "pessimistic", "regretful", "relieved", "remorseful", "resentful",
    "satisfied", "shocked", "sorrowful", "surprised", "terrified", "thrilled",
    "worried", "yearning", "delighted", "amazed", "upset", "elated", "cheerful",

    # Emotional states/descriptors
    "heart", "soul", "spirit", "mood", "feeling", "emotion", "sensation",
    "desire", "wish", "want", "need", "crave", "long", "yearn", "aspire",
    "dream", "hope", "fear", "worry", "concern", "care", "cherish", "treasure",
    "adore", "loathe", "despise", "detest", "dread", "enjoy", "appreciate",

    # Common emotional terms from samples
    "deep", "beautiful", "remarkable", "cherish", "wonderful", "terrible",
    "horrible", "amazing", "exciting", "devastating", "touching", "moving",
    "inspiring", "overwhelming", "comforting", "reassuring",

    # Relationship terms
    "friend", "family", "relationship", "connection", "bond", "attachment",
    "closeness", "intimacy", "trust", "betrayal", "loyalty", "devotion",

    # Intensity modifiers
    "very", "extremely", "incredibly", "deeply", "profoundly", "immensely",
    "utterly", "absolutely", "completely", "entirely", "truly", "really",

    # Emotional/relationship terms
    "believe", "personal", "relationship", "together", "shared",
    "memory", "childhood", "future"
]

# N-gram phrases to check (treated separately for efficiency)
logical_phrases = [
    "in conclusion", "research shows", "studies indicate", "data suggests",
    "evidence demonstrates", "logical conclusion", "scientific evidence",
    "statistically significant", "causal relationship", "objective analysis"
]

emotional_phrases = [
    "deep in my heart", "feel strongly", "emotionally affected", "breaks my heart",
    "brings me joy", "makes me happy", "deeply moved", "emotionally invested",
    "feel passionate", "care deeply", "think about"
]


class BooleanLogicExperiment:
    def __init__(self, model_name="distilgpt2", precision="auto", device=None, cache_dir=None):
        """
        Initialize the Boolean Logic Experiment with specified model

        Args:
            model_name: Name of model from MODEL_CONFIGS or HF model ID
            precision: Model precision (fp32, fp16, bf16, or auto)
            device: Device to run model on (cuda:0, cpu, or None for auto)
            cache_dir: Directory to cache models
        """
        self.model_name = model_name

        # Resolve model config
        if model_name in MODEL_CONFIGS:
            self.model_config = MODEL_CONFIGS[model_name]
            self.huggingface_id = self.model_config["huggingface_id"]
            self.tokenizer_id = self.model_config["tokenizer_id"]
        else:
            # Custom model ID
            self.model_config = {
                "name": model_name,
                "description": "Custom model",
                "huggingface_id": model_name,
                "tokenizer_id": model_name,
                "precision": precision if precision != "auto" else "fp32",
                "size_mb": 0,
            }
            self.huggingface_id = model_name
            self.tokenizer_id = model_name

        # Set up device
        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Determine precision
        if precision == "auto":
            self.precision = self.model_config["precision"]
        else:
            self.precision = precision

        # Set up cache directory
        self.cache_dir = cache_dir

        # Load tokenizer
        logger.info(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            cache_dir=self.cache_dir
        )

        # Log configuration
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Precision: {self.precision}")

        # Load model with appropriate precision
        logger.info(f"Loading model {self.huggingface_id}...")
        load_start = datetime.now()

        # Load in appropriate precision
        if self.precision == "fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.huggingface_id,
                torch_dtype=torch.float16,
                device_map=self.device if str(self.device) != "cpu" else None,
                cache_dir=self.cache_dir
            )
        elif self.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.huggingface_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device if str(self.device) != "cpu" else None,
                cache_dir=self.cache_dir
            )
        else:
            # Fall back to fp32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.huggingface_id,
                device_map=self.device if str(self.device) != "cpu" else None,
                cache_dir=self.cache_dir
            )

        # Move model to device if needed (for CPU)
        if str(self.device) == "cpu":
            self.model = self.model.to(self.device)

        load_time = (datetime.now() - load_start).total_seconds()
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        # Set up results directory
        self.results_dir = f"results/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        # Count dictionary terms
        self.logical_term_count = len(logical_words) + len(logical_phrases)
        self.emotional_term_count = len(emotional_words) + len(emotional_phrases)
        logger.info(f"Using expanded dictionaries:")
        logger.info(
            f"Logical terms: {self.logical_term_count} ({len(logical_words)} words + {len(logical_phrases)} phrases)")
        logger.info(
            f"Emotional terms: {self.emotional_term_count} ({len(emotional_words)} words + {len(emotional_phrases)} phrases)")

    def classify_token(self, token_text):
        """Classify token as logical, emotional, or neutral with confidence score"""
        token_lower = token_text.lower().strip()

        # Exact match for single words
        if token_lower in logical_words:
            return "logical", 1.0
        elif token_lower in emotional_words:
            return "emotional", 1.0

        # Check for phrases (n-grams)
        for phrase in logical_phrases:
            if phrase in token_lower:
                return "logical", 1.0

        for phrase in emotional_phrases:
            if phrase in token_lower:
                return "emotional", 1.0

        # Partial matching for words with length > 3
        for word in logical_words:
            if len(word) > 3 and (word in token_lower or token_lower in word):
                return "logical", 0.7

        for word in emotional_words:
            if len(word) > 3 and (word in token_lower or token_lower in word):
                return "emotional", 0.7

        return "neutral", 0.0

    def xor_gate(self, classifications):
        """Force either logical OR emotional but not both"""
        # Randomly select which mode to prefer for this generation
        desired_mode = "logical" if random.random() > 0.5 else "emotional"

        logic = {}
        for token, (cls, _) in classifications.items():
            logic[token] = (cls == desired_mode or cls == "neutral")
        return logic, desired_mode

    def or_gate(self, classifications):
        """Allow both logical AND emotional tokens (baseline)"""
        logic = {}
        for token, (cls, _) in classifications.items():
            logic[token] = (cls != "neutral")  # Prefer non-neutral tokens
        return logic, "mixed"

    def and_gate(self, classifications):
        """Only allow neutral tokens (for comparison)"""
        logic = {}
        for token, (cls, _) in classifications.items():
            logic[token] = (cls == "neutral")
        return logic, "neutral"

    def not_gate(self, classifications):
        """Suppress dominant token type"""
        # Count token types to determine dominant mode
        type_counts = {"logical": 0, "emotional": 0, "neutral": 0}
        for _, (cls, _) in classifications.items():
            type_counts[cls] += 1

        # Determine dominant non-neutral mode
        dominant = "logical" if type_counts["logical"] >= type_counts["emotional"] else "emotional"

        # Create logic that suppresses the dominant mode
        logic = {}
        for token, (cls, _) in classifications.items():
            logic[token] = (cls != dominant)

        return logic, f"not_{dominant}"

    def generate_with_gate(self, prompt, gate_function, tokens_to_generate=30, one_time=False):
        """Generate text with boolean logic control"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = inputs["input_ids"]

        # Track token classifications for analysis
        token_types = []
        token_transitions = []  # Track transitions between token types
        previous_token_type = None
        gate_mode = None

        for step in range(tokens_to_generate):
            with torch.no_grad():
                outputs = self.model(input_ids=generated_ids)
                next_token_logits = outputs.logits[:, -1, :]

            # Get top candidates
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)

            # Classify tokens
            token_classifications = {}
            for i in range(top_k):
                token_id = top_k_indices[0, i].item()
                token_text = self.tokenizer.decode([token_id])
                classification, confidence = self.classify_token(token_text)
                token_classifications[token_id] = (classification, confidence)

            # Decide whether to apply gating
            if one_time and step > 0:
                modified_logits = next_token_logits.clone()
                gate_mode = "none"
            else:
                modified_logic, gate_mode = gate_function(token_classifications)
                modified_logits = next_token_logits.clone()
                for token_id, allow in modified_logic.items():
                    if not allow:
                        modified_logits[0, token_id] -= 10.0  # Strong penalty

            # Sample from modified distribution
            next_token = torch.multinomial(
                torch.nn.functional.softmax(modified_logits, dim=-1),
                num_samples=1
            )

            # Track token type for analysis
            next_token_text = self.tokenizer.decode(next_token[0])
            token_class, _ = self.classify_token(next_token_text)
            token_types.append(token_class)

            # Track transitions between token types
            if previous_token_type is not None:
                transition = f"{previous_token_type}->{token_class}"
                token_transitions.append(transition)
            previous_token_type = token_class

            # Add to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text, token_types, token_transitions, gate_mode

    def calculate_metrics(self, token_types, token_transitions=None):
        """Calculate evaluation metrics from token type distribution"""
        total_tokens = len(token_types)
        logical_count = token_types.count("logical")
        emotional_count = token_types.count("emotional")
        neutral_count = token_types.count("neutral")

        logical_ratio = logical_count / total_tokens if total_tokens > 0 else 0
        emotional_ratio = emotional_count / total_tokens if total_tokens > 0 else 0
        neutral_ratio = neutral_count / total_tokens if total_tokens > 0 else 0

        # Mode skew measures how much the generation favors one mode over another
        mode_skew = abs(logical_ratio - emotional_ratio)

        # Calculate transition metrics if provided
        transition_metrics = {}
        if token_transitions:
            # Count each type of transition
            transition_counts = {}
            for transition in token_transitions:
                transition_counts[transition] = transition_counts.get(transition, 0) + 1

            # Calculate probabilities
            total_transitions = len(token_transitions)
            for transition, count in transition_counts.items():
                transition_metrics[transition] = count / total_transitions

            # Calculate coherence score (how often we stay in the same category)
            coherence_transitions = [t for t in token_transitions if t.startswith("logical->logical") or
                                     t.startswith("emotional->emotional") or
                                     t.startswith("neutral->neutral")]
            coherence_score = len(coherence_transitions) / total_transitions if total_transitions > 0 else 0
            transition_metrics["coherence_score"] = coherence_score

        return {
            "logical_ratio": logical_ratio,
            "emotional_ratio": emotional_ratio,
            "neutral_ratio": neutral_ratio,
            "mode_skew": mode_skew,
            "transition_metrics": transition_metrics
        }

    def run_quick_baseline(self, prompt):
        """Run a quick comparison between baseline and XOR gate for immediate feedback"""
        print(f"Running quick baseline test on prompt: '{prompt}'")

        # Generate with no gate (baseline)
        baseline_text, baseline_tokens, _, _ = self.generate_with_gate(prompt, self.or_gate)
        baseline_metrics = self.calculate_metrics(baseline_tokens)

        # Generate with XOR gate
        xor_text, xor_tokens, _, xor_mode = self.generate_with_gate(prompt, self.xor_gate)
        xor_metrics = self.calculate_metrics(xor_tokens)

        # Print results
        print("\n=== QUICK TEST RESULTS ===")
        print(
            f"BASELINE: {baseline_metrics['logical_ratio']:.2f} logical, {baseline_metrics['emotional_ratio']:.2f} emotional, {baseline_metrics['neutral_ratio']:.2f} neutral")
        print(
            f"XOR GATE ({xor_mode}): {xor_metrics['logical_ratio']:.2f} logical, {xor_metrics['emotional_ratio']:.2f} emotional, {xor_metrics['neutral_ratio']:.2f} neutral")
        print(f"DIFFERENCE IN MODE SKEW: {xor_metrics['mode_skew'] - baseline_metrics['mode_skew']:.2f}")

        print("\n=== GENERATED TEXT SAMPLES ===")
        print(f"BASELINE: {baseline_text}")
        print(f"XOR GATE: {xor_text}")

        return {
            "baseline_skew": baseline_metrics['mode_skew'],
            "xor_skew": xor_metrics['mode_skew'],
            "skew_difference": xor_metrics['mode_skew'] - baseline_metrics['mode_skew'],
            "baseline_neutral": baseline_metrics['neutral_ratio'],
            "xor_neutral": xor_metrics['neutral_ratio']
        }

    def run_experiment(self, prompts, gates, repetitions=3, tokens_to_generate=30):
        """Run full test suite and collect results"""
        results = []

        for prompt in prompts:
            for gate_name, gate_function in gates.items():
                for rep in tqdm(range(repetitions), desc=f"Testing {gate_name} gate"):
                    # Full gating
                    full_text, full_tokens, full_transitions, full_mode = self.generate_with_gate(
                        prompt, gate_function, tokens_to_generate=tokens_to_generate, one_time=False
                    )
                    full_metrics = self.calculate_metrics(full_tokens, full_transitions)
                    results.append({
                        **{"prompt": prompt, "gate": gate_name, "gate_mode": full_mode, "repetition": rep},
                        "generated_text": full_text,
                        "token_types": full_tokens,
                        "token_transitions": full_transitions,
                        **full_metrics
                    })
                    # One-time gating
                    once_text, once_tokens, once_transitions, once_mode = self.generate_with_gate(
                        prompt, gate_function, tokens_to_generate=tokens_to_generate, one_time=True
                    )
                    once_metrics = self.calculate_metrics(once_tokens, once_transitions)
                    results.append({
                        **{"prompt": prompt, "gate": f"{gate_name}_once", "gate_mode": once_mode, "repetition": rep},
                        "generated_text": once_text,
                        "token_types": once_tokens,
                        "token_transitions": once_transitions,
                        **once_metrics
                    })

        return pd.DataFrame(results)

    def analyze_results(self, results_df):
        """Perform statistical analysis and create visualizations"""
        # Group by gate type
        gate_groups = results_df.groupby('gate')

        # Compare logical/emotional ratios between gates
        xor_logical_ratios = results_df[results_df['gate'] == 'xor']['logical_ratio']
        or_logical_ratios = results_df[results_df['gate'] == 'or']['logical_ratio']

        # t-test to see if there's a significant difference
        t_stat, p_value = stats.ttest_ind(xor_logical_ratios, or_logical_ratios)

        print(f"T-test for logical ratio difference: t={t_stat:.4f}, p={p_value:.4f}")

        # Additional metrics by gate type
        print("\nMetrics by gate type:")
        print(gate_groups[['logical_ratio', 'emotional_ratio', 'neutral_ratio', 'mode_skew']].mean())

        # Mode skew comparison (higher = more polarized)
        skew_ttest = stats.ttest_ind(
            results_df[results_df['gate'] == 'xor']['mode_skew'],
            results_df[results_df['gate'] == 'or']['mode_skew']
        )

        print(f"\nT-test for mode skew difference: t={skew_ttest[0]:.4f}, p={skew_ttest[1]:.4f}")

        # Analyze token transitions
        print("\nAnalyzing token transitions:")
        transition_analysis = {}
        for gate in results_df['gate'].unique():
            gate_rows = results_df[results_df['gate'] == gate]
            all_transitions = []

            for _, row in gate_rows.iterrows():
                transition_metrics = row.get('transition_metrics', {})
                if 'coherence_score' in transition_metrics:
                    all_transitions.append(transition_metrics['coherence_score'])

            if all_transitions:
                transition_analysis[gate] = {
                    'mean_coherence': np.mean(all_transitions),
                    'std_coherence': np.std(all_transitions)
                }

        for gate, metrics in transition_analysis.items():
            print(f"  {gate.upper()}: Coherence = {metrics['mean_coherence']:.3f} ± {metrics['std_coherence']:.3f}")

        # Save metrics to file
        metrics_df = pd.DataFrame(index=gate_groups.groups.keys())

        # Add token type metrics
        for metric in ['logical_ratio', 'emotional_ratio', 'neutral_ratio', 'mode_skew']:
            metrics_df[metric] = gate_groups[metric].mean()

        # Add coherence metrics
        for gate, metrics in transition_analysis.items():
            metrics_df.loc[gate, 'coherence_mean'] = metrics['mean_coherence']
            metrics_df.loc[gate, 'coherence_std'] = metrics['std_coherence']

        # Save metrics
        metrics_df.to_csv(f"{self.results_dir}/metrics.csv")

        # Visualizations
        plt.figure(figsize=(15, 10))

        # Plot 1: Token type distribution by gate
        plt.subplot(2, 2, 1)
        token_avgs = gate_groups[['logical_ratio', 'emotional_ratio', 'neutral_ratio']].mean()
        token_avgs.plot(kind='bar', ax=plt.gca())
        plt.title('Token Type Distribution by Gate')
        plt.ylabel('Proportion of Tokens')

        # Plot 2: Mode skew comparison
        plt.subplot(2, 2, 2)
        sns.boxplot(x='gate', y='mode_skew', data=results_df)
        plt.title('Mode Skew by Gate Type')
        plt.ylabel('|Logical Ratio - Emotional Ratio|')

        # Plot 3: Coherence score by gate
        plt.subplot(2, 2, 3)
        coherence_data = []
        for gate in results_df['gate'].unique():
            gate_rows = results_df[results_df['gate'] == gate]
            for _, row in gate_rows.iterrows():
                transition_metrics = row.get('transition_metrics', {})
                if 'coherence_score' in transition_metrics:
                    coherence_data.append({
                        'gate': gate,
                        'coherence': transition_metrics['coherence_score']
                    })

        if coherence_data:
            coherence_df = pd.DataFrame(coherence_data)
            sns.boxplot(x='gate', y='coherence', data=coherence_df)
            plt.title('Coherence Score by Gate Type')
            plt.ylabel('Probability of Staying in Same Category')

        # Plot 4: Transition heatmap for XOR gate
        plt.subplot(2, 2, 4)
        transition_matrix = np.zeros((3, 3))
        categories = ['logical', 'emotional', 'neutral']
        category_to_idx = {cat: i for i, cat in enumerate(categories)}

        xor_rows = results_df[results_df['gate'] == 'xor']
        all_transitions = []

        for _, row in xor_rows.iterrows():
            transitions = row.get('token_transitions', [])
            for transition in transitions:
                from_cat, to_cat = transition.split('->')
                from_idx = category_to_idx[from_cat]
                to_idx = category_to_idx[to_cat]
                transition_matrix[from_idx, to_idx] += 1

        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

        sns.heatmap(transition_matrix, annot=True, fmt='.2f',
                    xticklabels=categories, yticklabels=categories,
                    cmap="YlGnBu")
        plt.title('XOR Gate Transition Probabilities')
        plt.xlabel('To Category')
        plt.ylabel('From Category')

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/boolean_logic_results.png")

        return {
            'logical_ratio_ttest': (t_stat, p_value),
            'skew_ttest': skew_ttest,
            'transition_analysis': transition_analysis,
            'gate_groups': gate_groups
        }

    def generate_evaluation_samples(self, prompts, num_samples=5):
        """Generate pairs of texts for human evaluation"""
        evaluation_pairs = []

        # Use a subset of prompts
        if len(prompts) <= num_samples:
            selected_prompts = prompts
        else:
            selected_prompts = random.sample(prompts, num_samples)

        for prompt in selected_prompts:
            # Generate with OR gate (baseline)
            or_text, _, _, _ = self.generate_with_gate(prompt, self.or_gate)

            # Generate with XOR gate
            xor_text, _, _, xor_mode = self.generate_with_gate(prompt, self.xor_gate)

            # Randomly order the pair to avoid bias
            if random.random() > 0.5:
                pair = {
                    "prompt": prompt,
                    "text_a": or_text,
                    "text_b": xor_text,
                    "a_type": "OR Gate",
                    "b_type": f"XOR Gate ({xor_mode})"
                }
            else:
                pair = {
                    "prompt": prompt,
                    "text_a": xor_text,
                    "text_b": or_text,
                    "a_type": f"XOR Gate ({xor_mode})",
                    "b_type": "OR Gate"
                }

            evaluation_pairs.append(pair)

        # Create a simple evaluation form
        print("\n=== HUMAN EVALUATION FORM ===")
        print("For each of the following text pairs, please rate which one:")
        print("1. Feels more logically consistent")
        print("2. Has better emotional coherence")
        print("3. Is more natural overall")
        print("Score each on a scale of 1-5 where 1=Text A much better, 3=Equal, 5=Text B much better\n")

        for i, pair in enumerate(evaluation_pairs):
            print(f"PAIR {i + 1}")
            print(f"Prompt: {pair['prompt']}")
            print(f"Text A: {pair['text_a']}")
            print(f"Text B: {pair['text_b']}")
            print("Logical consistency (1-5): ___")
            print("Emotional coherence (1-5): ___")
            print("Overall naturalness (1-5): ___")
            print()

        # Save evaluation form to file
        with open(f"{self.results_dir}/evaluation_form.txt", "w") as f:
            f.write("=== HUMAN EVALUATION FORM ===\n")
            f.write("For each of the following text pairs, please rate which one:\n")
            f.write("1. Feels more logically consistent\n")
            f.write("2. Has better emotional coherence\n")
            f.write("3. Is more natural overall\n")
            f.write("Score each on a scale of 1-5 where 1=Text A much better, 3=Equal, 5=Text B much better\n\n")

            for i, pair in enumerate(evaluation_pairs):
                f.write(f"PAIR {i + 1}\n")
                f.write(f"Prompt: {pair['prompt']}\n")
                f.write(f"Text A: {pair['text_a']}\n")
                f.write(f"Text B: {pair['text_b']}\n")
                f.write("Logical consistency (1-5): ___\n")
                f.write("Emotional coherence (1-5): ___\n")
                f.write("Overall naturalness (1-5): ___\n\n")

            # Add decoded pairs for analysis
            f.write("\n=== EVALUATION KEY (FOR ANALYSIS) ===\n")
            for i, pair in enumerate(evaluation_pairs):
                f.write(f"PAIR {i + 1}\n")
                f.write(f"Text A: {pair['a_type']}\n")
                f.write(f"Text B: {pair['b_type']}\n\n")

        print(f"\nEvaluation form saved to {self.results_dir}/evaluation_form.txt")
        input("Press Enter after completing the evaluation to continue...")

        return evaluation_pairs

    def run_full_experiment(self, prompts, repetitions=5, tokens_to_generate=30):
        """Run the full experiment pipeline"""
        # Define gates to test
        gates = {
            "xor": self.xor_gate,
            "or": self.or_gate,
            "and": self.and_gate,
            "not": self.not_gate
        }

        # First, run a quick baseline test for immediate feedback
        quick_results = self.run_quick_baseline(prompts[0])

        if quick_results["skew_difference"] > 0.1 or quick_results["baseline_neutral"] < 0.5:
            print("\nQuick test shows a meaningful difference! Proceeding with full experiment.")
        else:
            print("\nWarning: Quick test shows minimal difference between baseline and XOR gate.")
            proceed = input("Do you want to proceed with the full experiment? (y/n): ").lower()
            if proceed != 'y':
                print("Experiment aborted.")
                return

        # Run the experiment
        print(f"\nRunning Boolean Logic Gate experiment with {self.model_name}...")
        results = self.run_experiment(prompts, gates, repetitions=repetitions, tokens_to_generate=tokens_to_generate)

        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.results_dir}/boolean_logic_results_{timestamp}.csv"
        results.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        # Analyze and visualize results
        print("\nAnalyzing results...")
        stats_results = self.analyze_results(results)

        # Generate samples for human evaluation
        print("\nGenerating samples for human evaluation...")
        evaluation_pairs = self.generate_evaluation_samples(prompts)

        # Print representative examples
        print("\nSample outputs:")
        for gate in gates.keys():
            print(f"\n{gate.upper()} GATE EXAMPLES:")
            gate_samples = results[results['gate'] == gate]

            if len(gate_samples) >= 3:
                samples = gate_samples.sample(3)
            else:
                samples = gate_samples

            for _, row in samples.iterrows():
                print(f"Prompt: {row['prompt']}")
                print(f"Output: {row['generated_text']}")
                print(f"Gate mode: {row['gate_mode']}")
                print(f"Token types: {row['token_types'][:10]}...")  # Show first 10 tokens

                # Show token transitions
                transitions = row.get('token_transitions', [])
                if transitions:
                    print(f"Token transitions (first 5): {transitions[:5]}...")

                print(
                    f"Logical: {row['logical_ratio']:.2f}, Emotional: {row['emotional_ratio']:.2f}, Neutral: {row['neutral_ratio']:.2f}")

                # Show coherence score if available
                transition_metrics = row.get('transition_metrics', {})
                if 'coherence_score' in transition_metrics:
                    print(f"Coherence score: {transition_metrics['coherence_score']:.2f}")

                print("---")

        # Conclusion
        print("\nExperiment Conclusion:")
        if stats_results['skew_ttest'][1] < 0.05:
            print("The XOR gate produces significantly more polarized outputs than the OR gate,")
            print("confirming that boolean logic gates can effectively control language generation patterns.")
        else:
            print("No significant difference in polarization between gates was detected.")
            print("The boolean logic approach may need refinement to achieve consistent control.")

        # Get gate_groups from stats_results
        gate_groups = stats_results['gate_groups']

        # Neutral token analysis
        print("\nNeutral token analysis:")
        neutral_avgs = gate_groups['neutral_ratio'].mean()
        for gate, avg in neutral_avgs.items():
            print(f"  {gate.upper()}: {avg:.2f} neutral tokens")

        print("\nAdditional findings:")
        for gate, metrics in stats_results.get('transition_analysis', {}).items():
            if gate == 'xor' and 'mean_coherence' in metrics:
                if metrics['mean_coherence'] > 0.5:
                    print(
                        f"The XOR gate produces more coherent token sequences with a coherence score of {metrics['mean_coherence']:.2f}.")
                else:
                    print(
                        f"The XOR gate produces less coherent token sequences with a coherence score of {metrics['mean_coherence']:.2f}.")


from transformers import AutoTokenizer, AutoConfig

def download_model(model_name, cache_dir=None):
    """
    Download a model without loading it into memory.

    Args:
        model_name: Key of MODEL_CONFIGS to download.
        cache_dir: Optional directory to cache the model.
    Returns:
        True if download succeeded, False otherwise.
    """
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        return False

    config = MODEL_CONFIGS[model_name]

    try:
        # Download the tokenizer
        logger.info(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_id"],
            cache_dir=cache_dir
        )
        logger.info(f"Downloaded tokenizer for {model_name}")

        # Check if model config is accessible
        logger.info(f"Checking model {model_name}...")
        config_only = AutoConfig.from_pretrained(
            config["huggingface_id"],
            cache_dir=cache_dir
        )
        logger.info(f"Model {model_name} is accessible")

        return True
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        return False

def list_available_models():
    """List all predefined models and their descriptions to stdout."""
    print("\nAvailable models:")
    for name, config in MODEL_CONFIGS.items():
        print(f"- {name}: {config['description']} ({config['size_mb']}MB)")


def main():
    """CLI entrypoint: parse args, optionally download or list, then run experiments."""
    parser = argparse.ArgumentParser(description="Boolean Logic Gate Text Generation Experiment")
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="Model name from predefined configs or HuggingFace ID")
    parser.add_argument("--precision", type=str, default="auto",
                        choices=["auto", "fp32", "fp16", "bf16"],
                        help="Model precision")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run model on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache models")
    parser.add_argument("--reps", type=int, default=5,
                        help="Number of repetitions per prompt/gate")
    parser.add_argument("--tokens", type=int, default=30,
                        help="Number of tokens to generate per sample")
    parser.add_argument("--download", type=str, default=None,
                        help="Download a model without running experiments")
    parser.add_argument("--list-models", action="store_true",
                        help="List available predefined models")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--truth-table", action="store_true", help="Print all Boolean gate truth tables and exit")

    args = parser.parse_args()

    # Print truth tables and exit if requested
    if args.truth_table:
        generate_truth_tables()
        return

    # List models if requested
    if args.list_models:
        list_available_models()
        return

    # Download a model if requested
    if args.download:
        if args.download == "all":
            for model_name in MODEL_CONFIGS.keys():
                download_model(model_name, args.cache_dir)
        else:
            download_model(args.download, args.cache_dir)
        return

    # Interactive mode
    if args.interactive:
        models = list(MODEL_CONFIGS.keys())

        print("\nBoolean Logic Gate Text Generation Experiment")
        print("============================================")

        # Model selection
        print("\nAvailable models:")
        for i, model_name in enumerate(models):
            config = MODEL_CONFIGS[model_name]
            print(f"{i + 1}. {model_name}: {config['description']} ({config['size_mb']}MB)")

        print(f"{len(models) + 1}. Custom model (HuggingFace ID)")

        while True:
            try:
                model_choice = int(input("\nSelect model (number): "))
                if 1 <= model_choice <= len(models):
                    model_name = models[model_choice - 1]
                    break
                elif model_choice == len(models) + 1:
                    model_name = input("Enter HuggingFace model ID: ")
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

        # Precision selection
        if model_name in MODEL_CONFIGS:
            default_precision = MODEL_CONFIGS[model_name]["precision"]
            print(f"\nRecommended precision for {model_name}: {default_precision}")

        precision_options = ["auto", "fp32", "fp16", "bf16"]
        print("\nPrecision options:")
        for i, option in enumerate(precision_options):
            print(f"{i + 1}. {option}")

        while True:
            try:
                precision_choice = int(input("\nSelect precision (number): "))
                if 1 <= precision_choice <= len(precision_options):
                    precision = precision_options[precision_choice - 1]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

        # Device selection
        if torch.cuda.is_available():
            default_device = "cuda"
            device_options = ["cuda", "cpu"]
        else:
            default_device = "cpu"
            device_options = ["cpu"]

        print(f"\nAvailable devices: {', '.join(device_options)} (recommended: {default_device})")
        device = input(f"\nSelect device (press Enter for {default_device}): ")
        if not device:
            device = default_device

        # Repetitions
        reps = input("\nNumber of repetitions per prompt/gate (default: 5): ")
        reps = int(reps) if reps else 5

        # Number of tokens
        tokens = input("\nNumber of tokens to generate per sample (default: 30): ")
        tokens = int(tokens) if tokens else 30

        # Run experiment
        print(
            f"\nStarting experiment with {model_name} model, {precision} precision, {device} device, {reps} repetitions, {tokens} tokens...")

    else:
        # Command-line mode
        model_name = args.model
        precision = args.precision
        device = args.device
        reps = args.reps
        tokens = args.tokens

    # Define test prompts
    test_prompts = [
        "The scientists discovered that",
        "When I think about the future, I",
        "The most important thing to consider is",
        "Looking at the situation objectively",
        "Deep in my heart, I feel that"
    ]

    # Create experiment instance
    experiment = BooleanLogicExperiment(
        model_name=model_name,
        precision=precision,
        device=device,
        cache_dir=args.cache_dir
    )

    # Run full experiment
    experiment.run_full_experiment(test_prompts, repetitions=reps)

# --- Unit Tests for Pure Gates ---
import pytest

@pytest.mark.parametrize("a,b,expected", [(1,1,1),(1,0,0),(0,1,0),(0,0,0)])
def test_and_gate(a, b, expected):
    assert PURE_GATES["AND"][1](a, b) == expected

@pytest.mark.parametrize("a,b,expected", [(1,1,1),(1,0,1),(0,1,1),(0,0,0)])
def test_or_gate(a, b, expected):
    assert PURE_GATES["OR"][1](a, b) == expected

@pytest.mark.parametrize("a,b,expected", [(1,1,0),(1,0,1),(0,1,1),(0,0,1)])
def test_nand_gate(a, b, expected):
    assert PURE_GATES["NAND"][1](a, b) == expected

@pytest.mark.parametrize("a,b,expected", [(1,1,0),(1,0,0),(0,1,0),(0,0,1)])
def test_nor_gate(a, b, expected):
    assert PURE_GATES["NOR"][1](a, b) == expected

@pytest.mark.parametrize("a,b,expected", [(1,1,0),(1,0,1),(0,1,1),(0,0,0)])
def test_xor_gate(a, b, expected):
    assert PURE_GATES["XOR"][1](a, b) == expected

@pytest.mark.parametrize("a,b,expected", [(1,1,1),(1,0,0),(0,1,0),(0,0,1)])
def test_xnor_gate(a, b, expected):
    assert PURE_GATES["XNOR"][1](a, b) == expected

@pytest.mark.parametrize("a,expected", [(1,0),(0,1)])
def test_not_gate(a, expected):
    assert PURE_GATES["NOT"][1](a) == expected


if __name__ == "__main__":
    main()