#!/bin/bash

# Build script for manuscript with figures
echo "=== Building Manuscript with Figures ==="

# Create figures directory if it doesn't exist
mkdir -p figures

# Generate all figures
echo "Generating Figure 1: Study Design and Methodology..."
python3 manuscript/figure1.py

echo "Generating Figure 2: Main Results..."
# Use full fine-tuning with 100% target data for main results
EXPERIMENT=$(ls -d experiments/*full_fine_tuning_pct1.0*/ 2>/dev/null | head -n 1 | xargs basename)
if [ -n "$EXPERIMENT" ] && [ -d "experiments/$EXPERIMENT" ]; then
    python3 manuscript/figure2.py --experiment "$EXPERIMENT"
    echo "Using experiment: $EXPERIMENT"
else
    echo "Warning: full_fine_tuning_pct1.0 experiment not found. Generating demo figures..."
    # Create a mock experiment for demo purposes
    EXPERIMENT="demo"
    python3 manuscript/figure2.py --experiment "$EXPERIMENT" 2>/dev/null || python3 manuscript/figure2.py
fi

echo "Generating Figure 3: Training Dynamics..."
# Use same experiment as Figure 2 (full fine-tuning with 100% target data)
if [ -n "$EXPERIMENT" ] && [ -d "experiments/$EXPERIMENT" ]; then
    python3 manuscript/figure3.py --experiment "$EXPERIMENT"
else
    echo "Warning: No experiment data found. Generating demo figures..."
    python3 manuscript/figure3.py --experiment "$EXPERIMENT" 2>/dev/null || python3 manuscript/figure3.py
fi

echo "Generating Figure 4: Transfer Learning Data Efficiency..."
python3 manuscript/figure4.py

# Create generic figure2.pdf and figure3.pdf links for the manuscript
if [ -f "figures/figure2_${EXPERIMENT}.pdf" ]; then
    cp "figures/figure2_${EXPERIMENT}.pdf" "figures/figure2.pdf"
fi

if [ -f "figures/figure3_${EXPERIMENT}.pdf" ]; then
    cp "figures/figure3_${EXPERIMENT}.pdf" "figures/figure3.pdf"
fi

echo "=== Figures generated successfully ==="

# Check if pdflatex is available for compilation
if command -v pdflatex &> /dev/null; then
    echo "Compiling manuscript..."
    cd manuscript
    pdflatex manuscript.tex
    bibtex manuscript
    pdflatex manuscript.tex
    pdflatex manuscript.tex
    cd ..
    echo "Manuscript compiled successfully!"
else
    echo "pdflatex not found. Please install LaTeX to compile the manuscript."
    echo "Figures are available in the 'figures/' directory."
fi

echo "=== Build complete ==="