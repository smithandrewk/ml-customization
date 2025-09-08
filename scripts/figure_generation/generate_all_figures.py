#!/usr/bin/env python3
"""
Master Figure Generation Script
===============================

Generates all publication-ready figures for the personalized health monitoring paper.
Runs all individual figure generation scripts in sequence.

Usage:
    python scripts/figure_generation/generate_all_figures.py --results_dir results/lopo_6participants
    
Output:
    - figures/figure2_main_results.pdf/png - Main LOPO results
    - figures/figure3_training_curves.pdf/png - Training methodology
    - figures/summary_report.txt - Figure generation summary
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

def run_figure_script(script_name, results_dir, output_dir):
    """Run a figure generation script and capture results."""
    script_path = os.path.join('scripts', 'figure_generation', script_name)
    
    if not os.path.exists(script_path):
        return {
            'script': script_name,
            'status': 'error',
            'message': f'Script not found: {script_path}'
        }
    
    print(f"\n🎨 Running {script_name}...")
    
    cmd = ['python3', script_path, '--results_dir', results_dir, '--output_dir', output_dir]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            return {
                'script': script_name,
                'status': 'success',
                'message': 'Figure generated successfully',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            return {
                'script': script_name,
                'status': 'error',
                'message': f'Script failed with return code {result.returncode}',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    
    except subprocess.TimeoutExpired:
        return {
            'script': script_name,
            'status': 'timeout',
            'message': 'Script timed out after 5 minutes'
        }
    except Exception as e:
        return {
            'script': script_name,
            'status': 'error',
            'message': f'Exception occurred: {str(e)}'
        }

def check_prerequisites(results_dir):
    """Check if all required input files are available."""
    print(f"🔍 Checking prerequisites...")
    
    required_files = [
        'statistical_analysis.json',
        'participant_analysis.csv'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(results_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print(f"   Run 'python scripts/analyze_results.py --results_dir {results_dir}' first")
        return False
    
    print(f"✅ All prerequisites met")
    return True

def generate_summary_report(results, output_dir):
    """Generate a summary report of figure generation."""
    timestamp = datetime.now().isoformat()
    
    report = f"""Figure Generation Summary Report
=====================================

Generated: {timestamp}

FIGURE GENERATION RESULTS
========================

"""
    
    successful_figures = []
    failed_figures = []
    
    for result in results:
        script = result['script']
        status = result['status']
        
        if status == 'success':
            successful_figures.append(script)
            report += f"✅ {script}: SUCCESS\n"
        else:
            failed_figures.append(script)
            report += f"❌ {script}: {status.upper()}\n"
            report += f"   Error: {result['message']}\n"
        
        if result.get('stderr') and result['stderr'].strip():
            report += f"   Warnings: {result['stderr'][:200]}...\n"
        
        report += "\n"
    
    report += f"""
SUMMARY
=======

Total figures attempted: {len(results)}
Successful: {len(successful_figures)}
Failed: {len(failed_figures)}

Successful figures: {', '.join(successful_figures) if successful_figures else 'None'}
Failed figures: {', '.join(failed_figures) if failed_figures else 'None'}

GENERATED FILES
==============

The following figure files should be available in {output_dir}/:

Publication-ready (PDF):
"""
    
    # List expected output files
    expected_files = [
        'figure2_main_results.pdf',
        'figure3_training_curves.pdf'
    ]
    
    for file in expected_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            report += f"✅ {file}\n"
        else:
            report += f"❌ {file} (missing)\n"
    
    report += f"""
Presentation-ready (PNG):
"""
    
    png_files = [f.replace('.pdf', '.png') for f in expected_files]
    for file in png_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            report += f"✅ {file}\n"
        else:
            report += f"❌ {file} (missing)\n"
    
    report += f"""
NEXT STEPS
==========

If all figures generated successfully:
1. Review figures in {output_dir}/
2. Include in manuscript LaTeX document
3. Check figure quality and formatting
4. Generate supplementary figures if needed

If any figures failed:
1. Check error messages above
2. Ensure all prerequisites are met
3. Run individual figure scripts for debugging
4. Check data availability and format

USAGE IN MANUSCRIPT
==================

LaTeX figure inclusion example:

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\textwidth]{{figures/figure2_main_results.pdf}}
\\caption{{Leave-One-Participant-Out (LOPO) personalization results showing...}}
\\label{{fig:main_results}}
\\end{{figure}}
"""
    
    # Save report
    report_path = os.path.join(output_dir, 'figure_generation_summary.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Generate all publication figures')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing analysis results')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures (default: figures/)')
    parser.add_argument('--skip_checks', action='store_true',
                       help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.results_dir):
        print(f"❌ Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)
    
    print(f"🎨 GENERATING ALL PUBLICATION FIGURES")
    print(f"📂 Results directory: {args.results_dir}")
    print(f"🖼️  Output directory: {args.output_dir}")
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites(args.results_dir):
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define figures to generate
    figure_scripts = [
        'fig2_main_results.py',
        'fig3_training_curves.py'
    ]
    
    print(f"\n🚀 Starting figure generation...")
    print(f"   Figures to generate: {len(figure_scripts)}")
    
    # Generate figures
    results = []
    for script in figure_scripts:
        result = run_figure_script(script, args.results_dir, args.output_dir)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"   ✅ {script}: SUCCESS")
        else:
            print(f"   ❌ {script}: {result['status']} - {result['message']}")
    
    # Generate summary report
    print(f"\n📋 Generating summary report...")
    report_path = generate_summary_report(results, args.output_dir)
    
    # Final summary
    successful_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    
    print(f"\n{'='*50}")
    print(f"FIGURE GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"✅ Successful: {successful_count}/{total_count}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📋 Summary report: {report_path}")
    
    if successful_count == total_count:
        print(f"\n🎉 ALL FIGURES GENERATED SUCCESSFULLY!")
        print(f"   Ready for manuscript inclusion")
        print(f"   Check {args.output_dir}/ for PDF and PNG files")
    else:
        print(f"\n⚠️  Some figures failed to generate")
        print(f"   Check {report_path} for details")
        print(f"   Run individual scripts for debugging")
    
    # List generated files
    print(f"\n📄 Generated files:")
    if os.path.exists(args.output_dir):
        for file in sorted(os.listdir(args.output_dir)):
            if file.endswith(('.pdf', '.png')):
                print(f"   • {file}")
    
    # Exit with appropriate code
    if successful_count < total_count:
        sys.exit(1)

if __name__ == '__main__':
    main()