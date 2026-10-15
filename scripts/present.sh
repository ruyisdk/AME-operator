#!/usr/bin/env bash
# Live-demo driver. The accompanying speaker notes allocate 2-3 minutes.
set -euo pipefail
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$package_root"
if (( $# > 1 )); then
  echo "Usage: $0 [NEW_RESULT_DIRECTORY]" >&2
  exit 2
fi
output_dir=${1:-"$package_root/build/live-$(date +%Y%m%d-%H%M%S)-$$"}
printf '\nAME INT32 GEMM: RV64 / UDS32 / M16 / Acc4 / N=4\n'
printf '\n1. C/C++ intrinsic kernel\n'
cat examples/gemm_kernel.h
printf '\n2. Compile, disassemble, execute in Spike, verify every output\n'
python3 scripts/demo.py --language c --opt O2 --out "$output_dir"
printf '\n3. Selected instructions from the ELF just executed\n'
awk '/mls\.rm|mmov\.a\.m|mmulacc\.2d|mmov\.m\.a|mss\.rm/ { print; if (++n == 16) exit }' \
  "$output_dir/c-O2/disassembly.txt"
printf '\nResult: independent exact integer reference PASS\nEvidence: %s\n' "$output_dir"
