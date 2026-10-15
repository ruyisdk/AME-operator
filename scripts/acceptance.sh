#!/usr/bin/env bash
# Run the actual compiler -> Binutils -> Spike -> independent reference chain.
set -euo pipefail
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$package_root"
require_target=()
if [[ ${1:-} == --require-target ]]; then
  require_target=(--require-target)
  shift
fi
if (( $# > 1 )); then
  echo "Usage: $0 [--require-target] [NEW_RESULT_DIRECTORY]" >&2
  exit 2
fi
output_dir=${1:-"$package_root/build/acceptance-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$(dirname -- "$output_dir")"
mkdir "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
python3 scripts/check-host.py "${require_target[@]}" --output "$output_dir/host.json"
python3 scripts/demo.py --language all --opt all --negative-control \
  --out "$output_dir/demo"
printf '\nACCEPTANCE_PASS\nEvidence: %s\n' "$output_dir"
