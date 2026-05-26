#!/usr/bin/env bash
# Run all post-completion analyses after new jobs finish.
# Safe to run multiple times — scripts skip missing data gracefully.
#
# Usage: bash scripts/accv2026/run_post_completion_analyses.sh
#        bash scripts/accv2026/run_post_completion_analyses.sh --fde-routing
#        bash scripts/accv2026/run_post_completion_analyses.sh --all
set -euo pipefail

cd /data/wesleyferreiramaia/infoRates
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src

RUN_FDE=0
RUN_CASCADE=1   # on by default
RUN_KNAPSACK=1  # on by default
for arg in "$@"; do
  [[ "$arg" == "--fde-routing" ]] && RUN_FDE=1
  [[ "$arg" == "--no-cascade"  ]] && RUN_CASCADE=0
  [[ "$arg" == "--all"         ]] && RUN_FDE=1
done

echo "============================================================"
echo "Post-completion analyses — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo ""
echo "[1/6] Compiling paper tables (08_compile_paper_results.py)..."
python scripts/accv2026/08_compile_paper_results.py

echo ""
echo "[2/6] Computing TDS correlation (07_dataset_temporal_demand.py)..."
python scripts/accv2026/07_dataset_temporal_demand.py \
  --output evaluations/accv2026/dataset_temporal_demand.csv 2>/dev/null || true

echo ""
echo "[3/6] Plotting paper figures (09_plot_paper_figures.py)..."
python scripts/accv2026/09_plot_paper_figures.py

echo ""
echo "[3b/6] Routing comparison figure (14_plot_routing_comparison.py)..."
python scripts/accv2026/14_plot_routing_comparison.py 2>/dev/null || true

echo ""
echo "[4/6] Per-class temporal analysis (10_per_class_temporal_analysis.py)..."
python scripts/accv2026/10_per_class_temporal_analysis.py 2>/dev/null | grep -v DeprecationWarning || true

if [[ $RUN_CASCADE -eq 1 ]]; then
  echo ""
  echo "[5/6] Confidence cascade routing (12_confidence_cascade.py)..."
  python scripts/accv2026/12_confidence_cascade.py \
    --eval-base  evaluations/accv2026/fixed_budget \
    --output-dir evaluations/accv2026/confidence_cascade \
    --datasets ssv2 ucf101 hmdb51 diving48 epic 2>/dev/null || true
fi

if [[ $RUN_KNAPSACK -eq 1 ]]; then
  echo ""
  echo "[6/6] Knapsack + confidence allocator (13_knapsack_confidence.py)..."
  python scripts/accv2026/13_knapsack_confidence.py \
    --eval-base  evaluations/accv2026/fixed_budget \
    --output-dir evaluations/accv2026/knapsack_confidence \
    --datasets ssv2 ucf101 hmdb51 diving48 epic 2>/dev/null || true
fi

if [[ $RUN_FDE -eq 1 ]]; then
  echo ""
  echo "[FDE] Running FDE adaptive routing for all available samples CSVs..."
  for eval_dir in evaluations/accv2026/fixed_budget/*/; do
    for samples_csv in "${eval_dir}"*_fixed_budget_samples.csv; do
      [[ -f "$samples_csv" ]] || continue
      fde_cache="${eval_dir}fde_cache.csv"
      [[ -f "$fde_cache" ]] || continue
      echo "  FDE routing: $samples_csv"
      python scripts/accv2026/06_fde_adaptive_routing.py \
        --samples-csv "$samples_csv" \
        --fde-cache   "$fde_cache" \
        --output-dir  "$eval_dir" \
        --workers 8 2>/dev/null || echo "    [SKIP] failed"
    done
  done
fi

echo ""
echo "============================================================"
echo "Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Paper results:    evaluations/accv2026/paper_results/"
echo "Cascade results:  evaluations/accv2026/confidence_cascade/"
echo "Knapsack results: evaluations/accv2026/knapsack_confidence/"
N_FILES=$(ls evaluations/accv2026/paper_results/*.csv \
             evaluations/accv2026/paper_results/figures/*.pdf \
             evaluations/accv2026/confidence_cascade/*.csv \
             evaluations/accv2026/knapsack_confidence/*.csv 2>/dev/null | wc -l)
echo "${N_FILES} output files ready"
echo "============================================================"
