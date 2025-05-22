#!/bin/bash
# Run script for Boolean Logic Text Generation testing

# Create a directory for prompts if it doesn't exist
mkdir -p prompts

# Check if the prompt file already exists
if [ ! -f prompts/test_prompts.txt ]; then
  echo "Creating test prompts file..."
  cat > prompts/test_prompts.txt << EOF
# Boolean Logic Test Prompt Suite
# One prompt per line - used as input for the test suite

# CATEGORY: Neutral prompts (balanced/ambiguous starting points)
The situation can be described as
When considering the matter,
Looking at this from a different perspective,
In this particular scenario,
Let's examine the following:

# CATEGORY: Logical-leaning prompts
The scientific evidence suggests that
Based on a rational analysis,
The data clearly indicates
The logical conclusion would be
According to the research findings,

# CATEGORY: Emotional-leaning prompts
When I think about my childhood,
The most beautiful experience was
It breaks my heart when
I've always dreamed about
Nothing makes me happier than

# CATEGORY: Domain-specific prompts
The mathematical formula demonstrates
The poetry evokes feelings of
The philosophical argument states
The relationship between them was
The political implications include

# CATEGORY: Mixed prompts (deliberate logical/emotional tension)
While the facts suggest otherwise, I feel
Despite my emotional reaction, logically
Both the heart and mind agree that
Balancing reason and passion, we
Objectively speaking, the situation makes me
EOF
  echo "Test prompts file created at prompts/test_prompts.txt"
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
  echo "Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if not already installed
if ! pip freeze | grep -q transformers; then
  echo "Installing dependencies..."
  pip install torch transformers pandas matplotlib seaborn scipy tqdm
  echo "Dependencies installed."
fi

# Function to run a specific test configuration
run_test() {
  local test_name=$1
  local gates=$2
  local reps=$3
  local tokens=$4

  echo "Running test: $test_name"
  echo "Gates: $gates, Repetitions: $reps, Tokens: $tokens"

  python boolean_logic_larger_model.py \
    --prompts prompts/test_prompts.txt \
    --gates $gates \
    --reps $reps \
    --tokens $tokens \
    --output "results/$test_name"

  echo "Test $test_name completed."
  echo "Results available in results/$test_name_*/"
}

# Create results directory
mkdir -p results

# Run tests
echo "Starting Boolean Logic Gate testing..."

# Test 1: Quick baseline test (all gates, fewer repetitions)
run_test "quick_baseline" "baseline,xor,or,and,not,nand" 2 20

# Test 2: XOR vs Baseline comparison (focused test with more repetitions)
run_test "xor_vs_baseline" "xor,baseline" 5 30

# Test 3: All gates comparison (comprehensive with more repetitions)
run_test "all_gates_comparison" "baseline,xor,or,and,not,nand" 3 30

# Test 4: Long generation test (fewer gates but longer outputs)
run_test "long_generation" "xor,or,baseline" 3 50

echo "All tests completed."
echo "View HTML reports in the results directories for detailed analysis."

# Deactivate virtual environment
deactivate