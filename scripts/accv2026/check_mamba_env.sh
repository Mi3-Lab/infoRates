#!/usr/bin/env bash
# Diagnose why mamba_ssm fails to import. The published VideoMamba sweeps ran on
# 2026-06-16, so the environment worked then; something in the system CUDA has
# changed since. Run on a compute node, which is where those sweeps executed.
cd /scratch/wesleyferreiramaia/infoRates
VM=/data/wesleyferreiramaia/infoRates/.venv_mamba/bin/python
echo "=== node: $(hostname) ==="
echo "--- system libcudart ---"
find /usr/local /opt /usr/lib64 -maxdepth 4 -name "libcudart.so*" 2>/dev/null | head -5
echo "--- cuda modules ---"
module avail 2>&1 | grep -io "cuda/[0-9.]*" | sort -u | head -5
echo "--- plain import ---"
$VM -c "import mamba_ssm; print('MAMBA OK')" 2>&1 | tail -2
echo "--- with cu12 runtime on LD_LIBRARY_PATH ---"
L=$(dirname "$(find /data/wesleyferreiramaia/infoRates/.venv_mamba -name 'libcudart.so.12' 2>/dev/null | head -1)")
echo "    lib dir: $L"
LD_LIBRARY_PATH="$L:${LD_LIBRARY_PATH:-}" $VM -c "import mamba_ssm; print('MAMBA OK via LD_LIBRARY_PATH')" 2>&1 | tail -2
echo "--- symlink 12->13 in a scratch dir (reversible probe) ---"
S=/scratch/wesleyferreiramaia/infoRates/_cudashim; mkdir -p "$S"
ln -sf "$L/libcudart.so.12" "$S/libcudart.so.13" 2>/dev/null
LD_LIBRARY_PATH="$S:$L:${LD_LIBRARY_PATH:-}" $VM -c "import mamba_ssm; print('MAMBA OK via shim')" 2>&1 | tail -2
