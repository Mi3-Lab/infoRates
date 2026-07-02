#!/usr/bin/env bash
# Launch InfoRates dashboard
# Usage: bash dashboard/run_dashboard.sh [port]

cd "$(dirname "$0")/.."
PORT="${1:-8501}"

source .venv/bin/activate

echo "=== InfoRates Dashboard ==="
echo "URL: http://localhost:${PORT}"
echo "For SSH tunnel: ssh -L ${PORT}:localhost:${PORT} <cluster_user>@<cluster_host>"
echo ""

echo "YOU SHOULD NOT BE RUNNING WEB APPS ON LOGIN NODES!!"

#streamlit run dashboard/app.py \
#    --server.port "${PORT}" \
#    --server.headless true \
#    --server.address 0.0.0.0 \
#    --browser.gatherUsageStats false \
#    --theme.base light \
#    --theme.primaryColor "#2c7be5"
