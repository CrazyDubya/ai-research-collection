import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Load model + tokenizer
model_name = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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


# Enhanced token classification with support for phrases
def classify_token(token_text):
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
def generate_with_gate(prompt, gate_function, tokens_to_generate=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = inputs["input_ids"]

    # Track token classifications for analysis
    token_types = []
    token_transitions = []  # Track transitions between token types
    previous_token_type = None
    gate_mode = None

    for _ in range(tokens_to_generate):
        outputs = model(input_ids=generated_ids)
        next_token_logits = outputs.logits[:, -1, :]

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

        # Sample from modified distribution
        next_token = torch.multinomial(
            torch.nn.functional.softmax(modified_logits, dim=-1),
            num_samples=1
        )

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
def run_quick_baseline(prompt):
    """Run a quick comparison between baseline and XOR gate for immediate feedback"""
    print(f"Running quick baseline test on prompt: '{prompt}'")

    # Generate with no gate (baseline)
    baseline_text, baseline_tokens, _, _ = generate_with_gate(prompt, or_gate)
    baseline_metrics = calculate_metrics(baseline_tokens)

    # Generate with XOR gate
    xor_text, xor_tokens, _, xor_mode = generate_with_gate(prompt, xor_gate)
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
def run_experiment(prompts, gates, repetitions=3):
    results = []

    for prompt in prompts:
        for gate_name, gate_function in gates.items():
            for rep in range(repetitions):
                generated_text, token_types, token_transitions, gate_mode = generate_with_gate(prompt, gate_function)
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
        'transition_analysis': transition_analysis
    }


# Function to generate human evaluation samples
def generate_evaluation_samples(prompts, num_samples=5):
    """Generate pairs of texts for human evaluation"""
    evaluation_pairs = []

    # Use a subset of prompts
    selected_prompts = random.sample(prompts, min(num_samples, len(prompts)))

    for prompt in selected_prompts:
        # Generate with OR gate (baseline)
        or_text, _, _, _ = generate_with_gate(prompt, or_gate)

        # Generate with XOR gate
        xor_text, _, _, xor_mode = generate_with_gate(prompt, xor_gate)

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

    return evaluation_pairs


# Main experimental function
def main():
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
        "xor": xor_gate,
        "or": or_gate,
        "and": and_gate,
        "not": not_gate
    }

    # Print dictionary statistics
    logical_count = len(logical_words) + len(logical_phrases)
    emotional_count = len(emotional_words) + len(emotional_phrases)
    print(f"Using expanded dictionaries:")
    print(f"Logical terms: {logical_count} ({len(logical_words)} words + {len(logical_phrases)} phrases)")
    print(f"Emotional terms: {emotional_count} ({len(emotional_words)} words + {len(emotional_phrases)} phrases)")

    # First, run a quick baseline test for immediate feedback
    quick_results = run_quick_baseline(test_prompts[0])

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
    results = run_experiment(test_prompts, gates, repetitions=5)

    # Save raw results
    results.to_csv("boolean_logic_results_expanded.csv", index=False)

    # Analyze and visualize results
    print("\nAnalyzing results...")
    stats_results = analyze_results(results)
    # Add to main() function
    print("\nNeutral token analysis:")
    gate_groups = results.groupby('gate')
    neutral_avgs = gate_groups['neutral_ratio'].mean()
    for gate, avg in neutral_avgs.items():
        print(f"  {gate.upper()}: {avg:.2f} neutral tokens")

    # Generate samples for human evaluation
    print("\nGenerating samples for human evaluation...")
    evaluation_pairs = generate_evaluation_samples(test_prompts)

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


if __name__ == "__main__":
    main()