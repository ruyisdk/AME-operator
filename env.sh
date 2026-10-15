#!/usr/bin/env bash
# Source this file from Bash: source ./env.sh
AME_TUTORIAL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export AME_TUTORIAL_ROOT
export PATH="$AME_TUTORIAL_ROOT/toolchain/bin:$AME_TUTORIAL_ROOT/spike/bin:$PATH"
