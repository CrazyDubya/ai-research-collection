import early_blocker

import sys
import os
import builtins

os.environ["TINYTROUPE_DISABLE"] = "1"

_real_import = builtins.__import__

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "tinytroupe" in name.lower():
        raise ImportError(f"TinyTroupe forcibly disabled (blocked {name})")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = safe_import

print("[Early Blocker] TinyTroupe disabled.")
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import gc
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face authentication token
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
use_auth = True if HF_TOKEN else False
if use_auth:
    logger.info("Using HUGGING_FACE_HUB_TOKEN for authentication")
else:
    logger.warning("HUGGING_FACE_HUB_TOKEN not found, proceeding without authentication")


def clear_gpu_memory():
    """Clear CUDA cache to free up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Boolean Logic Gates for Language Models")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="Model name or path")
    parser.add_argument("--precision", type=str, default="full",
                        choices=["full", "half"],
                        help="Model precision (full or half)")
    parser.add_argument("--tokens", type=int, default=30,
                        help="Number of tokens to generate")
    parser.add_argument("--repetitions", type=int, default=3,
                        help="Number of repetitions per prompt/gate")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--quick_test", action="store_true",
                        help="Run only a quick test")
    return parser.parse_args()


def load_model(model_name, precision="full", device: str | None = None, cache_dir: str | None = None):
    """Load model with specified precision, device selection, and optional cache dir"""
    logger.info(f"Loading model {model_name} with {precision} precision...")

    # Determine device
    if device is None:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        elif torch.cuda.is_available():
            device_obj = torch.device("cuda")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    logger.info(f"Using device: {device_obj}")

    # Prepare kwargs for from_pretrained
    load_kwargs = {
        "cache_dir": cache_dir,
        "use_auth_token": HF_TOKEN if use_auth else None,
    }

    # Precision handling (Full or Half)
    if precision not in ("full", "half"):
        logger.error(f"Unsupported precision '{precision}'. Only 'full' or 'half' are supported.")
        raise ValueError(f"Unsupported precision: {precision}")

    dtype = torch.float16 if precision == "half" else torch.float32
    if device_obj.type in ("cuda", "mps"):
        load_kwargs.update({
            "torch_dtype": dtype,
            "device_map": "auto",
        })
    else:
        # CPU only
        load_kwargs.update({
            "torch_dtype": torch.float32, # Always use float32 on CPU for stability
        })

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Ensure CPU device placement if requested
    if device_obj.type == "cpu":
        model.to("cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN if use_auth else None, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device_obj


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


# --- Boolean Bitmask Logic for Token Types ---
# Bitmask mapping: logical=0b100, emotional=0b010, neutral=0b001
TOKEN_TYPE_BITMASK = {
    "logical": 0b100,
    "emotional": 0b010,
    "neutral": 0b001
}
BITMASK_TO_TYPE = {0b100: "logical", 0b010: "emotional", 0b001: "neutral"}

# Helper: classify token and return bitmask

def classify_token_bitmask(token_text):
    token_lower = token_text.lower().strip()
    if token_lower in logical_words:
        return 0b100
    elif token_lower in emotional_words:
        return 0b010
    for phrase in logical_phrases:
        if phrase in token_lower:
            return 0b100
    for phrase in emotional_phrases:
        if phrase in token_lower:
            return 0b010
    for word in logical_words:
        if len(word) > 3 and (word in token_lower or token_lower in word):
            return 0b100
    for word in emotional_words:
        if len(word) > 3 and (word in token_lower or token_lower in word):
            return 0b010
    return 0b001

# --- Boolean Gate Functions using Bitmask Logic ---
# Each gate receives a dict {token_id: bitmask}

def xor_gate_bitmask(classifications):
    """Allow logical XOR emotional (but not both, not neutral)."""
    logic = {}
    for token_id, bitmask in classifications.items():
        logic[token_id] = (bitmask == 0b100 or bitmask == 0b010)  # Only one bit set
    return logic, "xor"

def or_gate_bitmask(classifications):
    """Allow logical OR emotional (not neutral)."""
    logic = {}
    for token_id, bitmask in classifications.items():
        logic[token_id] = bool(bitmask & (0b100 | 0b010))  # Any non-neutral
    return logic, "or"

def and_gate_bitmask(classifications):
    """Allow only tokens that are both logical AND emotional (rare)."""
    logic = {}
    for token_id, bitmask in classifications.items():
        logic[token_id] = (bitmask == (0b100 | 0b010))
    return logic, "and"

def not_gate_bitmask(classifications):
    """Allow only neutral tokens (NOT logical NOR emotional)."""
    logic = {}
    for token_id, bitmask in classifications.items():
        logic[token_id] = (bitmask == 0b001)
    return logic, "not"

# Example: Add a NAND gate (not AND)
def nand_gate_bitmask(classifications):
    logic = {}
    for token_id, bitmask in classifications.items():
        logic[token_id] = not (bitmask == (0b100 | 0b010))
    return logic, "nand"

# --- Update generate_with_gate to use bitmask logic ---
@torch.no_grad()
def generate_with_gate(prompt, gate_function, model, tokenizer, device, tokens_to_generate=30,
                       temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    token_types = []
    token_transitions = []
    previous_token_type = None
    gate_mode = None

    for _ in range(tokens_to_generate):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask
        )
        next_token_logits = outputs.logits[:, -1, :]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        top_k = 50
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)

        # Classify tokens using bitmask
        token_classifications = {}
        for i in range(top_k):
            token_id = top_k_indices[0, i].item()
            token_text = tokenizer.decode([token_id])
            bitmask = classify_token_bitmask(token_text)
            token_classifications[token_id] = bitmask

        # Apply Boolean gate
        modified_logic, gate_mode = gate_function(token_classifications)

        modified_logits = next_token_logits.clone()
        for token_id, allow in modified_logic.items():
            if not allow:
                modified_logits[0, token_id] -= 10.0

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(modified_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            modified_logits[0, indices_to_remove] = -float('Inf')

        probs = torch.nn.functional.softmax(modified_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Track token type (string) for analysis
        next_token_text = tokenizer.decode(next_token[0])
        bitmask = classify_token_bitmask(next_token_text)
        token_class = BITMASK_TO_TYPE.get(bitmask, "neutral")
        token_types.append(token_class)

        if previous_token_type is not None:
            transition = f"{previous_token_type}->{token_class}"
            token_transitions.append(transition)
        previous_token_type = token_class

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=torch.long, device=device)
        ], dim=-1)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, token_types, token_transitions, gate_mode

# ---
# To add a new Boolean gate: define a function like above, using bitmask logic, and add it to the 'gates' dict in main().



# Logic Gates - implementing just XOR and OR for clear comparison
def xor_gate(classifications):
    """Force either logical OR emotional but not both"""
    # Randomly select which mode to prefer for this generation
    desired_mode = "logical" if random.random() > 0.5 else "emotional"

    logic = {}
    for token, (cls, _) in classifications.items():
        logic[token] = (cls == desired_mode or cls == "neutral")
    return logic, desired_mode


def or_gate(classifications):
    """Allow both logical AND emotional tokens (baseline)"""
    logic = {}
    for token, (cls, _) in classifications.items():
        logic[token] = (cls != "neutral")  # Prefer non-neutral tokens
    return logic, "mixed"


def and_gate(classifications):
    """Only allow neutral tokens (for comparison)"""
    logic = {}
    for token, (cls, _) in classifications.items():
        logic[token] = (cls == "neutral")
    return logic, "neutral"


def not_gate(classifications):
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


# Generate text with boolean logic control
@torch.no_grad()  # Disable gradient calculation for inference
def generate_with_gate(prompt, gate_function, model, tokenizer, device, tokens_to_generate=30,
                       temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Track token classifications for analysis
    token_types = []
    token_transitions = []  # Track transitions between token types
    previous_token_type = None
    gate_mode = None

    for _ in range(tokens_to_generate):
        # Get model outputs for current sequence
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask
        )
        next_token_logits = outputs.logits[:, -1, :]

        # Apply temperature sampling
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Get top candidates
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)

        # Classify tokens
        token_classifications = {}
        for i in range(top_k):
            token_id = top_k_indices[0, i].item()
            token_text = tokenizer.decode([token_id])
            classification, confidence = classify_token(token_text)
            token_classifications[token_id] = (classification, confidence)

        # Apply the selected boolean logic gate
        modified_logic, gate_mode = gate_function(token_classifications)

        # Apply penalties to filtered tokens
        modified_logits = next_token_logits.clone()
        for token_id, allow in modified_logic.items():
            if not allow:
                modified_logits[0, token_id] -= 10.0  # Strong penalty

        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(modified_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            modified_logits[0, indices_to_remove] = -float('Inf')

        # Sample from modified distribution
        probs = torch.nn.functional.softmax(modified_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Track token type for analysis
        next_token_text = tokenizer.decode(next_token[0])
        token_class, _ = classify_token(next_token_text)
        token_types.append(token_class)

        # Track transitions between token types
        if previous_token_type is not None:
            transition = f"{previous_token_type}->{token_class}"
            token_transitions.append(transition)
        previous_token_type = token_class

        # Add to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), dtype=torch.long, device=device)
        ], dim=-1)

    # Clear CUDA cache after each generation to prevent memory buildup
    if device.type == "cuda":
        torch.cuda.empty_cache()

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, token_types, token_transitions, gate_mode


# Evaluation metrics
def calculate_metrics(token_types, token_transitions=None):
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


# Quick baseline test function for immediate feedback
def run_quick_baseline(prompt, model, tokenizer, device):
    """Run a quick comparison between baseline and XOR gate for immediate feedback"""
    print(f"Running quick baseline test on prompt: '{prompt}'")

    # Generate with no gate (baseline)
    baseline_text, baseline_tokens, _, _ = generate_with_gate(
        prompt, or_gate, model, tokenizer, device
    )
    baseline_metrics = calculate_metrics(baseline_tokens)

    # Generate with XOR gate
    xor_text, xor_tokens, _, xor_mode = generate_with_gate(
        prompt, xor_gate, model, tokenizer, device
    )
    xor_metrics = calculate_metrics(xor_tokens)

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


# Experimental runner
def run_experiment(prompts, gates, model, tokenizer, device, repetitions=3, tokens_per_generation=30):
    results = []

    for prompt in prompts:
        for gate_name, gate_function in gates.items():
            print(f"Testing gate: {gate_name} on prompt: '{prompt}'")
            for rep in range(repetitions):
                print(f"  Repetition {rep + 1}/{repetitions}")
                generated_text, token_types, token_transitions, gate_mode = generate_with_gate(
                    prompt, gate_function, model, tokenizer, device, tokens_per_generation
                )
                metrics = calculate_metrics(token_types, token_transitions)

                results.append({
                    "prompt": prompt,
                    "gate": gate_name,
                    "gate_mode": gate_mode,
                    "repetition": rep,
                    "generated_text": generated_text,
                    "token_types": token_types,
                    "token_transitions": token_transitions,
                    **metrics
                })

    return pd.DataFrame(results)


# Statistical analysis and visualization
def analyze_results(results_df):
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
    plt.savefig('boolean_logic_results.png')

    return {
        'logical_ratio_ttest': (t_stat, p_value),
        'skew_ttest': skew_ttest,
        'transition_analysis': transition_analysis,
        'gate_groups': gate_groups  # Return gate_groups for use in main
    }


# Function to generate human evaluation samples
def generate_evaluation_samples(prompts, model, tokenizer, device, num_samples=5):
    """Generate pairs of texts for human evaluation"""
    evaluation_pairs = []

    # Use a subset of prompts
    selected_prompts = random.sample(prompts, min(num_samples, len(prompts)))

    for prompt in selected_prompts:
        # Generate with OR gate (baseline)
        or_text, _, _, _ = generate_with_gate(prompt, or_gate, model, tokenizer, device)

        # Generate with XOR gate
        xor_text, _, _, xor_mode = generate_with_gate(prompt, xor_gate, model, tokenizer, device)

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

    # Wait for human evaluation
    input("\nPress Enter after completing the evaluation to continue...")

    return evaluation_pairs


# Main experimental function
def main():
    # Parse arguments
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer, device = load_model(args.model, args.precision)

    # Set of diverse prompts for testing
    test_prompts = [
        "The scientists discovered that",
        "When I think about the future, I",
        "The most important thing to consider is",
        "Looking at the situation objectively",
        "Deep in my heart, I feel that"
    ]

    # Gates to test
    gates = {
        "xor": xor_gate_bitmask,
        "or": or_gate_bitmask,
        "and": and_gate_bitmask,
        "not": not_gate_bitmask,
        # Add more gates here, e.g. "nand": nand_gate_bitmask
    }

    # Print dictionary statistics
    logical_count = len(logical_words) + len(logical_phrases)
    emotional_count = len(emotional_words) + len(emotional_phrases)
    print(f"Using expanded dictionaries:")
    print(f"Logical terms: {logical_count} ({len(logical_words)} words + {len(logical_phrases)} phrases)")
    print(f"Emotional terms: {emotional_count} ({len(emotional_words)} words + {len(emotional_phrases)} phrases)")

    # Save experiment configuration
    config = {
        "model": args.model,
        "precision": args.precision,
        "tokens_per_generation": args.tokens,
        "repetitions": args.repetitions,
        "logical_count": logical_count,
        "emotional_count": emotional_count,
        "timestamp": timestamp
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        import json
        json.dump(config, f, indent=2)

    # First, run a quick baseline test for immediate feedback
    quick_results = run_quick_baseline(test_prompts[0], model, tokenizer, device)

    # If only running quick test
    if args.quick_test:
        print("\nQuick test completed. Exiting as requested.")
        return

    if quick_results["skew_difference"] > 0.1 or quick_results["baseline_neutral"] < 0.5:
        print("\nQuick test shows a meaningful difference! Proceeding with full experiment.")
    else:
        print("\nWarning: Quick test shows minimal difference between baseline and XOR gate.")
        proceed = input("Do you want to proceed with the full experiment? (y/n): ").lower()
        if proceed != 'y':
            print("Experiment aborted.")
            return

    # Run the experiment
    print("\nRunning Boolean Logic Gate experiment...")
    results = run_experiment(
        test_prompts, gates, model, tokenizer, device,
        repetitions=args.repetitions,
        tokens_per_generation=args.tokens
    )

    # Save raw results
    results_path = os.path.join(run_dir, f"boolean_logic_results_{timestamp}.csv")
    results.to_csv(results_path, index=False)
    print(f"Raw results saved to {results_path}")

    # Analyze and visualize results
    print("\nAnalyzing results...")
    stats_results = analyze_results(results)
    gate_groups = stats_results['gate_groups']

    # Generate samples for human evaluation
    print("\nGenerating samples for human evaluation...")
    evaluation_pairs = generate_evaluation_samples(test_prompts, model, tokenizer, device)

    # Save evaluation pairs
    eval_path = os.path.join(run_dir, f"evaluation_pairs_{timestamp}.json")
    with open(eval_path, "w") as f:
        eval_data = [{
            "prompt": pair["prompt"],
            "text_a": pair["text_a"],
            "text_b": pair["text_b"],
            "a_type": pair["a_type"],
            "b_type": pair["b_type"]
        } for pair in evaluation_pairs]
        json.dump(eval_data, f, indent=2)

    # Print representative examples
    print("\nSample outputs:")
    for gate in gates.keys():
        print(f"\n{gate.upper()} GATE EXAMPLES:")
        samples = results[results['gate'] == gate].sample(min(3, len(results[results['gate'] == gate])))
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

    # Clean up
    clear_gpu_memory()
    print(f"\nAll results saved to {run_dir}")
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()