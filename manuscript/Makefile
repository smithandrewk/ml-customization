# Makefile for automated manuscript generation
# Usage: make pdf EXPERIMENT=<experiment_name>

# Default experiments (can be overridden)
EXPERIMENT ?= b512_aug_patience25_full_fine_tuning_pct1.0_20250930_085838
TARGET_ONLY_EXP ?= b512_aug_patience25_target_only_pct1.0_20250930_085817
FULL_FT_EXP ?= b512_aug_patience25_full_fine_tuning_pct1.0_20250930_085838

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Main manuscript file
MAIN = manuscript
TEX_FILE = $(MAIN).tex
PDF_FILE = $(MAIN).pdf

# Auto-generated files
RESULTS_FILE = results.tex
ABSTRACT_STATS_FILE = abstract_stats.tex
GENERATED_FILES = generated_tables.tex standalone_tables.tex

.PHONY: all pdf clean results help

# Default target
all: pdf

# Generate combined results from target-only and full fine-tuning experiments
results:
	@echo "Generating combined results from target-only: $(TARGET_ONLY_EXP) and full FT: $(FULL_FT_EXP)"
	python3 generate_combined_results.py --target-only $(TARGET_ONLY_EXP) --full-ft $(FULL_FT_EXP)
	python3 generate_abstract_stats.py --target-only $(TARGET_ONLY_EXP) --full-ft $(FULL_FT_EXP)
	@echo "Generated files: $(RESULTS_FILE), $(ABSTRACT_STATS_FILE)"

# Compile PDF (with auto-generated results)
pdf: results
	@echo "Compiling manuscript with experiment: $(EXPERIMENT)"
	$(LATEX) $(TEX_FILE)
	@if grep -q "\\\\cite" $(MAIN).aux; then \
		echo "Running bibtex..."; \
		$(BIBTEX) $(MAIN); \
		$(LATEX) $(TEX_FILE); \
	fi
	$(LATEX) $(TEX_FILE)
	@echo "PDF generated: $(PDF_FILE)"

# Compile PDF without regenerating results
pdf-only:
	@echo "Compiling manuscript (using existing results.tex)"
	$(LATEX) $(TEX_FILE)
	@if grep -q "\\\\cite" $(MAIN).aux; then \
		echo "Running bibtex..."; \
		$(BIBTEX) $(MAIN); \
		$(LATEX) $(TEX_FILE); \
	fi
	$(LATEX) $(TEX_FILE)

# Clean build files
clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.fls *.fdb_latexmk
	rm -f $(RESULTS_FILE) $(ABSTRACT_STATS_FILE) $(GENERATED_FILES)
	@echo "Cleaned build files"

# Clean everything including PDF
clean-all: clean
	rm -f $(PDF_FILE)
	@echo "Cleaned all files"

# Generate results for different experiment
update-experiment:
	@if [ -z "$(EXPERIMENT)" ]; then \
		echo "Error: Please specify EXPERIMENT=<experiment_name>"; \
		exit 1; \
	fi
	@echo "Updating manuscript for experiment: $(EXPERIMENT)"
	sed -i 's/\\newcommand{\\experimentname}{.*}/\\newcommand{\\experimentname}{$(EXPERIMENT)}/' $(TEX_FILE)
	$(MAKE) results
	@echo "Manuscript updated for experiment: $(EXPERIMENT)"

# Show help
help:
	@echo "Manuscript build system"
	@echo ""
	@echo "Usage:"
	@echo "  make pdf                                    - Generate PDF with default experiments (combined target-only + full FT)"
	@echo "  make pdf TARGET_ONLY_EXP=<exp1> FULL_FT_EXP=<exp2> - Generate PDF with specific experiments"
	@echo "  make results                                - Generate combined results.tex and abstract_stats.tex"
	@echo "  make pdf-only                               - Compile PDF without regenerating results"
	@echo "  make clean                                  - Clean build files"
	@echo "  make clean-all                              - Clean everything including PDF"
	@echo ""
	@echo "Default experiments:"
	@echo "  TARGET_ONLY_EXP = $(TARGET_ONLY_EXP)"
	@echo "  FULL_FT_EXP     = $(FULL_FT_EXP)"
	@echo ""
	@echo "Standalone tools:"
	@echo "  python3 generate_combined_results.py --target-only <exp1> --full-ft <exp2> - Generate combined comparison"
	@echo "  python3 generate_abstract_stats.py --target-only <exp1> --full-ft <exp2>   - Generate abstract statistics"
	@echo "  python3 generate_results.py <name>                                         - Generate single experiment results (legacy)"
	@echo ""
	@echo "Examples:"
	@echo "  make pdf"
	@echo "  make pdf TARGET_ONLY_EXP=b256_target_exp FULL_FT_EXP=b256_full_ft_exp"
	@echo "  make results TARGET_ONLY_EXP=my_target_exp FULL_FT_EXP=my_full_ft_exp"