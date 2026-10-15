#!/usr/bin/env bash
# Rebuild the frozen simulator in a new directory, then run selected host tests.
set -euo pipefail
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 SOURCE_DIR NEW_BUILD_DIR [INSTALL_PREFIX]" >&2
  exit 2
fi
source_dir=$(realpath -- "$1")
build_dir=$(realpath -m -- "$2")
install_prefix=$(realpath -m -- "${3:-$package_root/spike}")
jobs=${JOBS:-6}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive" >&2; exit 2; }
[[ -f "$source_dir/configure" ]] || { echo "Missing frozen Spike source" >&2; exit 2; }
if [[ -e "$build_dir" ]] && [[ -n $(ls -A -- "$build_dir") ]]; then
  echo "Build directory must be new or empty: $build_dir" >&2
  exit 2
fi
mkdir -p -- "$build_dir" "$install_prefix/bin" "$install_prefix/share/licenses/spike"
printf 'Source: %s\nBuild: %s\nInstall: %s\nJobs: %s\n' "$source_dir" "$build_dir" "$install_prefix" "$jobs"

g++ --version
make --version
dtc --version
cd -- "$build_dir"
# Boost supplies the optional socket interface; local interactive debugging
# and AME execution need no Boost installation.
"$source_dir/configure" --prefix="$install_prefix" \
  --without-boost --without-boost-asio --without-boost-regex \
  'CXXFLAGS=-O2 -g0' 'CFLAGS=-O2 -g0'
# The upstream Makefile expands LINK once and then evals its link recipes.
# Four dollar signs survive these two make expansions as one literal dollar.
runtime_libdir="'\$\$\$\$ORIGIN/../lib'"
make -j"$jobs" "install_libs_dir=$runtime_libdir" spike
install -m755 spike "$install_prefix/bin/spike"
install -m644 "$source_dir/LICENSE" "$install_prefix/share/licenses/spike/LICENSE"
ldd "$install_prefix/bin/spike"
if ldd "$install_prefix/bin/spike" | grep -Eq 'not found|libriscv|libfesvr|libsoftfloat|libdisasm|libfdt'; then
  echo 'Unexpected external simulator library; installation needs repair.' >&2
  exit 1
fi
readelf -d "$install_prefix/bin/spike" | grep -F '[$ORIGIN/../lib]'
tests=(ame-unit-config-and-ownership-utst ame-elementwise-convert-utst \
  ame-elementwise-arith-utst ame-matmul-gemm-utst \
  ame-matmul-transpose-accumulate-utst)
make -j"$jobs" "install_libs_dir=$runtime_libdir" "${tests[@]}"
for test_name in "${tests[@]}"; do
  printf '\nRunning %s\n' "$test_name"
  timeout 120 "./$test_name"
done
"$install_prefix/bin/spike" --help || [[ $? == 1 ]]
printf '\nInstalled Spike and passed all %s selected host test programs.\n' "${#tests[@]}"
