#!/usr/bin/env bash
#SBATCH --job-name=build-mamba-ssm
#SBATCH --partition=cenvalarc.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=evaluations/accv2026/logs/%x-%j.out
#SBATCH --error=evaluations/accv2026/logs/%x-%j.err
set -euo pipefail
cd /data/wesleyferreiramaia/infoRates

echo "=== Build mamba-ssm for .venv_mamba ==="
echo "Node: $(hostname)  |  $(date)"
nvidia-smi | head -5 || true

source .venv_mamba/bin/activate
pip install wheel --quiet  # required by causal-conv1d setup.py
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

# The cluster has only CUDA 13.x nvcc, but PyTorch was compiled with CUDA 12.8.
# causal-conv1d rejects nvcc when major version != PyTorch's CUDA major.
# Workaround: create a fake nvcc wrapper that reports version 12.8 for the
# version check, but delegates all actual compilation to the real nvcc 13.x.
# The compiled extension uses PyTorch's CUDA 12.8 headers (via include_paths),
# so the generated code only calls CUDA 12.8 APIs and links safely against
# PyTorch's bundled libcudart.so.12.
REAL_NVCC=$(find /usr/local/cuda-13.*/bin/nvcc /usr/local/cuda-13/bin/nvcc 2>/dev/null | head -1 || which nvcc 2>/dev/null || echo "")
if [[ -z "$REAL_NVCC" ]]; then
    echo "[ERROR] No nvcc found at all" >&2
    exit 1
fi
echo "Real nvcc: ${REAL_NVCC} ($(${REAL_NVCC} --version | grep release))"

FAKE_CUDA_DIR="$(mktemp -d)/fake_cuda_12.8"
mkdir -p "${FAKE_CUDA_DIR}/bin"
PYTORCH_CUDA="${PYTORCH_CUDA:-12.8}"
FAKE_NVCC="${FAKE_CUDA_DIR}/bin/nvcc"

cat > "${FAKE_NVCC}" << 'FAKENVCC'
#!/usr/bin/env bash
# Fake nvcc: reports CUDA 12.8 for version checks, delegates compilation to real nvcc 13.x.
# Also filters out GPU architectures not supported by CUDA 13.x (sm_5x and sm_6x were
# dropped in CUDA 13). mamba-ssm setup.py adds compute_53/62/70/72 which cause:
#   nvcc fatal: Unsupported gpu architecture 'compute_53'
REAL_NVCC="__REAL_NVCC__"
PYTORCH_CUDA="__PYTORCH_CUDA__"

if [[ "${1:-}" == "-V" || "${1:-}" == "--version" || "$*" == "-V" ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2024 NVIDIA Corporation"
    echo "Built on Mon_Apr__3_17:16:06_PDT_2024"
    echo "Cuda compilation tools, release ${PYTORCH_CUDA}, V${PYTORCH_CUDA}.89"
    echo "Build cuda_${PYTORCH_CUDA}.r${PYTORCH_CUDA}/compiler.fake_0"
else
    # Filter -gencode pairs whose arch < compute_75 (not supported by CUDA 13.x)
    args=("$@")
    filtered=()
    i=0
    while [[ $i -lt ${#args[@]} ]]; do
        arg="${args[$i]}"
        if [[ "$arg" == "-gencode" && $((i+1)) -lt ${#args[@]} ]]; then
            next="${args[$((i+1))]}"
            if [[ "$next" =~ ^arch=compute_([0-9]+), ]]; then
                cap="${BASH_REMATCH[1]}"
                if [[ "$cap" -lt 75 ]]; then
                    i=$((i+2))  # skip this -gencode arch=... pair
                    continue
                fi
            fi
        fi
        filtered+=("$arg")
        i=$((i+1))
    done
    exec "${REAL_NVCC}" "${filtered[@]}"
fi
FAKENVCC
# Substitute the real values into the wrapper (single-quoted heredoc needs sed)
sed -i "s|__REAL_NVCC__|${REAL_NVCC}|g; s|__PYTORCH_CUDA__|${PYTORCH_CUDA}|g" "${FAKE_NVCC}"
chmod +x "${FAKE_NVCC}"

# Symlink the real CUDA 13.x include dir — basic CUDA headers (cuda_runtime.h,
# vector_types.h, etc.) are the same between 12.x and 13.x for operations that
# causal-conv1d uses. The compiled .so links against PyTorch's libcudart.so.12.
REAL_CUDA_DIR=$(dirname "$(dirname "$REAL_NVCC")")
ln -sf "${REAL_CUDA_DIR}/include" "${FAKE_CUDA_DIR}/include"
ln -sf "${REAL_CUDA_DIR}/lib64"   "${FAKE_CUDA_DIR}/lib64" 2>/dev/null || true

export CUDA_HOME="${FAKE_CUDA_DIR}"
echo "CUDA_HOME=${CUDA_HOME} (fake nvcc wrapping ${REAL_NVCC})"
echo "  CUDA include: $(readlink ${CUDA_HOME}/include)"
"${FAKE_NVCC}" -V

pip install opencv-python-headless --quiet  # required by train_videomamba.py
echo "Installing causal-conv1d (full output)..."
MAX_JOBS=4 pip install causal-conv1d --no-build-isolation 2>&1 || { echo "causal-conv1d install FAILED"; exit 1; }

echo "Installing mamba-ssm (full output)..."
MAX_JOBS=4 pip install mamba-ssm --no-build-isolation 2>&1 || { echo "mamba-ssm install FAILED"; exit 1; }

# Patch Python 3.10+ type hint syntax (X | None) to work on Python 3.9.
# causal-conv1d 1.6.x uses `X | None` union type in torch.library.custom_op functions.
# PyTorch's infer_schema evaluates these annotations at runtime, so we must replace
# 'X | None' → 'Optional[X]' directly (not via __future__ which would stringify them).
echo "Patching Python 3.10+ type hints for Python 3.9 compatibility..."
SITE_PKG=".venv_mamba/lib/python3.9/site-packages"
python3 - << 'PYEOF'
import re, sys
from pathlib import Path

site = Path(".venv_mamba/lib/python3.9/site-packages")
for pkg in ["causal_conv1d", "mamba_ssm"]:
    for f in sorted((site / pkg).rglob("*.py") if (site / pkg).exists() else []):
        text = f.read_text()
        if "| None" not in text and "| None" not in text:
            continue
        # Add Optional, Tuple imports after first 'import torch' line
        new = re.sub(
            r'^(import torch\b)',
            r'\1\nfrom typing import Optional, Tuple',
            text, count=1, flags=re.MULTILINE
        )
        # Replace 'X | None' → 'Optional[X]' (handle nested brackets too)
        new = re.sub(r'([\w\.]+(?:\[[\w\.,\s]+\])?)\s*\|\s*None', r'Optional[\1]', new)
        # Replace 'tuple[...]' → 'Tuple[...]' for Python 3.9
        new = re.sub(r'\btuple\[', 'Tuple[', new)
        if new != text:
            f.write_text(new)
            print(f"  Patched: {f}")
PYEOF

echo ""
echo "=== Verification ==="
python3 -c "
import causal_conv1d
print('causal_conv1d OK:', causal_conv1d.__version__)
import mamba_ssm
print('mamba_ssm OK:', mamba_ssm.__version__)
from mamba_ssm.modules.mamba_simple import Mamba
import torch
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 10, 64).cuda()
y = m(x)
print('Mamba forward OK: input', x.shape, '-> output', y.shape)
"
echo "=== mamba-ssm build COMPLETE ==="
