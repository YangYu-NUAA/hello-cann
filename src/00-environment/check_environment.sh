#!/usr/bin/env bash

set -u

section() {
  printf '\n===== %s =====\n' "$1"
}

find_cann_env() {
  if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    printf '%s\n' "${ASCEND_HOME_PATH}/set_env.sh"
    return
  fi

  if [ -f "${HOME}/Ascend/ascend-toolkit/set_env.sh" ]; then
    printf '%s\n' "${HOME}/Ascend/ascend-toolkit/set_env.sh"
    return
  fi

  find /usr/local/Ascend "${HOME}/Ascend" -maxdepth 3 -name set_env.sh 2>/dev/null \
    | awk '/\/cann-[^/]+\/set_env\.sh$/ || /\/ascend-toolkit\/set_env\.sh$/ {print; exit}'
}

section "System"
uname -a
grep '^PRETTY_NAME=' /etc/os-release 2>/dev/null || true
python3 --version

section "NPU"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info

  npu_id=$(npu-smi info 2>/dev/null | awk -F'|' '/Ascend/ {gsub(/^ +| +$/, "", $2); split($2, field, " "); print field[1]; exit}')
  if [ -n "${npu_id}" ]; then
    npu-smi info -t board -i "${npu_id}" 2>/dev/null \
      | grep -E 'NPU ID|Product Name|Software Version|Firmware Version|Compatibility|Chip Count' || true
  fi
else
  echo "npu-smi: not found"
fi

section "Driver"
if [ -f /usr/local/Ascend/driver/version.info ]; then
  grep -E '^(Version|Innerversion|package_version)=' /usr/local/Ascend/driver/version.info
else
  echo "/usr/local/Ascend/driver/version.info: not found"
fi

section "CANN"
cann_env=$(find_cann_env)
if [ -z "${cann_env}" ]; then
  echo "CANN set_env.sh: not found"
else
  echo "set_env.sh: ${cann_env}"
  echo "resolved: $(readlink -f "${cann_env}")"
  # shellcheck disable=SC1090
  source "${cann_env}"
  echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
  echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"

  install_info="${ASCEND_HOME_PATH:-}/aarch64-linux/ascend_toolkit_install.info"
  if [ -f "${install_info}" ]; then
    grep -E '^(package_name|version|innerversion|arch|os|path)=' "${install_info}"
  fi
fi

section "HCCL"
if [ -n "${ASCEND_HOME_PATH:-}" ]; then
  find "${ASCEND_HOME_PATH}" -name 'libhccl.so*' -print 2>/dev/null
fi
python3 - <<'PY'
import ctypes.util

print("ctypes hccl:", ctypes.util.find_library("hccl"))
PY

section "PyTorch and torch_npu"
python3 - <<'PY'
import traceback

try:
    import torch
    import torch_npu

    print("torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print("npu available:", torch.npu.is_available())
    print("npu count:", torch.npu.device_count())

    x = torch.ones((2, 3), device="npu")
    y = x + 1
    print("device:", y.device)
    print("result:", y.cpu())
except Exception:
    traceback.print_exc()
PY

section "Optional tools"
if command -v docker >/dev/null 2>&1; then
  docker --version
else
  echo "docker: not found"
fi

python3 - <<'PY'
try:
    import transformers
    print("transformers:", transformers.__version__)
except ModuleNotFoundError:
    print("transformers: not installed (install it in 01 Inference)")
PY
