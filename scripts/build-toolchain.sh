#!/usr/bin/env bash
# Build only from frozen source copies; never reuse development build trees.
set -euo pipefail
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_ROOT=${SOURCE_ROOT:-"$bundle_dir/sources"}
BUILD_ROOT=${BUILD_ROOT:-"${TMPDIR:-/tmp}/ame-summit-rebuild-$UID"}
PREFIX=${PREFIX:-"$bundle_dir/toolchain"}
JOBS=${JOBS:-12}
TARGET=riscv64-unknown-elf
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] && (( JOBS <= 12 )) || {
  echo 'JOBS must be an integer between 1 and 12.' >&2; exit 2;
}
for component in binutils gcc; do
  [[ -f "$SOURCE_ROOT/$component/configure" ]] || {
    echo "Missing $SOURCE_ROOT/$component/configure; extract the frozen source archive first." >&2
    exit 2
  }
done
mkdir -p "$PREFIX" "$BUILD_ROOT"
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd)
BUILD_ROOT=$(cd "$BUILD_ROOT" && pwd)
PREFIX=$(cd "$PREFIX" && pwd)
for component in binutils gcc; do
  if [[ -e "$BUILD_ROOT/$component" ]] && [[ -n $(ls -A -- "$BUILD_ROOT/$component") ]]; then
    echo "Refusing to reuse a nonempty build directory: $BUILD_ROOT/$component" >&2
    echo 'Set BUILD_ROOT to a new directory for this frozen source and prefix.' >&2
    exit 2
  fi
done
export PATH="$PREFIX/bin:$PATH"
export LC_ALL=C
export CFLAGS=${CFLAGS:--O2 -g0}
export CXXFLAGS=${CXXFLAGS:--O2 -g0}

run() {
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

mkdir -p "$BUILD_ROOT/binutils"
cd "$BUILD_ROOT/binutils"
run "$SOURCE_ROOT/binutils/configure" \
    --target="$TARGET" --prefix="$PREFIX" \
    --disable-multilib --disable-nls --disable-werror \
    --disable-gdb --disable-sim --disable-gprofng \
    --enable-plugins --with-system-zlib
run make -j"$JOBS" all-binutils all-gas all-ld
run make install-binutils install-gas install-ld

mkdir -p "$BUILD_ROOT/gcc"
cd "$BUILD_ROOT/gcc"
run "$SOURCE_ROOT/gcc/configure" \
    --target="$TARGET" --prefix="$PREFIX" \
    --enable-languages=c,c++,lto \
    --disable-multilib --disable-bootstrap --disable-shared \
    --disable-threads --disable-nls --disable-werror \
    --disable-libssp --disable-libquadmath --disable-libgomp \
    --disable-libatomic --without-headers --with-newlib \
    --with-arch=rv64im_zicsr --with-abi=lp64 \
    --with-system-zlib --with-static-standard-libraries=yes
run make -j"$JOBS" all-gcc
run make install-gcc
printf 'Compiler ready: %s/bin/%s-gcc\n' "$PREFIX" "$TARGET"
run make -j"$JOBS" all-target-libgcc
run make install-target-libgcc

"$PREFIX/bin/$TARGET-gcc" --version
"$PREFIX/bin/$TARGET-g++" --version
test -f "$("$PREFIX/bin/$TARGET-gcc" -print-file-name=include)/riscv_ztt.h"
test -f "$("$PREFIX/bin/$TARGET-gcc" -print-libgcc-file-name)"
printf 'Toolchain installed in %s\n' "$PREFIX"
