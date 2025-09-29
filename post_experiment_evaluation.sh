#!/bin/bash
#
# Post-Experiment Evaluation Script
#
# This script should be run after completing training experiments to:
# 1. Compute test set metrics for all experiments
# 2. Generate manuscript results and figures
# 3. Create a summary report
#
# Usage:
#   ./post_experiment_evaluation.sh [experiment_name]
#   ./post_experiment_evaluation.sh --all           # Evaluate all experiments
#   ./post_experiment_evaluation.sh <experiment>    # Evaluate specific experiment
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
EXPERIMENT_NAME=""
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            EXPERIMENT_NAME="--all"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all              Evaluate all experiments"
            echo "  --device DEVICE    GPU device to use (default: cuda:0)"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --all"
            echo "  $0 --device cuda:1"
            echo "  $0 alpha  # Evaluate specific experiment named 'alpha'"
            exit 0
            ;;
        *)
            EXPERIMENT_NAME="$1"
            shift
            ;;
    esac
done

# Default to --all if no argument provided
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="--all"
fi

print_header "Post-Experiment Evaluation Pipeline"

echo "Configuration:"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Device: ${DEVICE}"
echo ""

# Check if experiments directory exists
if [ ! -d "experiments" ]; then
    print_error "experiments/ directory not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if evaluation script exists
if [ ! -f "evaluate_on_test_set.py" ]; then
    print_error "evaluate_on_test_set.py not found!"
    exit 1
fi

# Step 1: Evaluate test sets
print_header "Step 1: Computing Test Set Metrics"

if [ "$EXPERIMENT_NAME" = "--all" ]; then
    python3 evaluate_on_test_set.py --all --device "$DEVICE"
else
    python3 evaluate_on_test_set.py --experiment "$EXPERIMENT_NAME" --device "$DEVICE"
fi

if [ $? -eq 0 ]; then
    print_success "Test set evaluation completed"
else
    print_error "Test set evaluation failed"
    exit 1
fi

# Step 2: Generate manuscript results
print_header "Step 2: Generating Manuscript Results"

# Find the latest experiment if --all was used
if [ "$EXPERIMENT_NAME" = "--all" ]; then
    # Get the most recent experiment directory
    LATEST_EXP=$(ls -t experiments/ | head -1)
    print_warning "Using latest experiment for manuscript: $LATEST_EXP"
    MANUSCRIPT_EXP="$LATEST_EXP"
else
    MANUSCRIPT_EXP="$EXPERIMENT_NAME"
fi

# Generate results.tex and abstract_stats.tex
cd manuscript
python3 generate_results.py "$MANUSCRIPT_EXP"

if [ $? -eq 0 ]; then
    print_success "Manuscript results generated (results.tex, abstract_stats.tex)"
else
    print_error "Failed to generate manuscript results"
    cd ..
    exit 1
fi

cd ..

# Step 3: Generate figures
print_header "Step 3: Generating Figures"

echo "Generating Figure 2 (Main Results)..."
python3 manuscript/figure2.py --experiment "$MANUSCRIPT_EXP" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Figure 2 generated"
else
    print_warning "Figure 2 generation had warnings (check output)"
fi

echo "Generating Figure 3 (Training Dynamics)..."
python3 manuscript/figure3.py --experiment "$MANUSCRIPT_EXP" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Figure 3 generated"
else
    print_warning "Figure 3 generation had warnings (check output)"
fi

echo "Generating Figure 4 (Data Efficiency)..."
python3 manuscript/figure4.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Figure 4 generated"
else
    print_warning "Figure 4 generation had warnings (check output)"
fi

# Copy figures to standard names
cd figures
cp "figure2_${MANUSCRIPT_EXP}.pdf" figure2.pdf 2>/dev/null || true
cp "figure3_${MANUSCRIPT_EXP}.pdf" figure3.pdf 2>/dev/null || true
# figure4.pdf already has standard name
cd ..

# Step 4: Generate summary report
print_header "Step 4: Generating Summary Report"

REPORT_FILE="evaluation_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$REPORT_FILE" << EOF
================================================================================
Post-Experiment Evaluation Report
Generated: $(date)
================================================================================

Experiment: $MANUSCRIPT_EXP
Device: $DEVICE

================================================================================
Test Set Evaluation Results
================================================================================

EOF

# Extract summary from test evaluation log if it exists
if [ -f "test_evaluation_log.txt" ]; then
    grep -A 10 "SUMMARY" test_evaluation_log.txt >> "$REPORT_FILE" 2>/dev/null || true
fi

cat >> "$REPORT_FILE" << EOF

================================================================================
Files Generated
================================================================================

Manuscript Results:
  - manuscript/results.tex
  - manuscript/abstract_stats.tex

Figures:
  - figures/figure2.pdf
  - figures/figure3.pdf
  - figures/figure4.pdf

================================================================================
Next Steps
================================================================================

1. Review results:
   - Check manuscript/results.tex for performance metrics
   - View figures in figures/ directory

2. Compile manuscript:
   - cd manuscript && pdflatex manuscript.tex

3. If results look good, commit:
   - git add experiments/$MANUSCRIPT_EXP
   - git add manuscript/results.tex manuscript/abstract_stats.tex
   - git add figures/figure*.pdf
   - git commit -m "Add test set evaluation for $MANUSCRIPT_EXP"

4. For full n=15 experiments:
   - Update train.py: participants = ['tonmoy','alsaad','anam','asfik','ejaz','iftakhar','unk1','dennis',...]
   - Re-run training for all folds
   - Re-run this script: ./post_experiment_evaluation.sh --all

================================================================================
EOF

print_success "Summary report saved to: $REPORT_FILE"

# Display key findings
print_header "Summary"

echo "Experiments evaluated: $MANUSCRIPT_EXP"
echo ""
echo "Generated files:"
echo "  - Test metrics: experiments/*/fold*/metrics.json"
echo "  - Results: manuscript/results.tex"
echo "  - Figures: figures/figure2.pdf, figure3.pdf, figure4.pdf"
echo "  - Report: $REPORT_FILE"
echo ""

print_success "Post-experiment evaluation complete!"

echo ""
echo "To view results:"
echo "  cat $REPORT_FILE"
echo "  cd manuscript && pdflatex manuscript.tex && open manuscript.pdf"
echo ""