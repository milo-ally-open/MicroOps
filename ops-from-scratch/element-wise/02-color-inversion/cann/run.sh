#!/usr/bin/env bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
cann_root="${ASCEND_HOME_DIR:-/usr/local/Ascend/ascend-toolkit/latest}"
kernel_impl_dir="${cann_root}/compiler/tikcpp/tikcfw/impl/dav_m300"

if [[ ! -d "${kernel_impl_dir}" ]]; then
    echo "CANN Ascend C target dav_m300 (Ascend 310B) is unavailable in ${cann_root}." >&2
    echo "Expected implementation headers at: ${kernel_impl_dir}" >&2
    echo "Use a CANN toolkit that supports the board's Ascend 310B target; no build or benchmark was run." >&2
    exit 2
fi

mkdir -p "${build_dir}"

ccec -x cce "${script_dir}/solve.cpp" -x c++ "${script_dir}/bench.cpp" \
    --cce-aicore-arch=dav-m300 \
    -mllvm -cce-aicore-function-stack-size=16000 \
    -mllvm -cce-aicore-fp-ceiling=2 \
    -mllvm -cce-aicore-record-overflow=false \
    -std=c++17 -O2 \
    -I"${cann_root}/runtime/include" \
    -I"${cann_root}/compiler/tikcpp/tikcfw" \
    -I"${cann_root}/compiler/tikcpp/tikcfw/impl" \
    -I"${cann_root}/compiler/tikcpp/tikcfw/interface" \
    -L"${cann_root}/atc/lib64" \
    -L"${cann_root}/runtime/lib64" \
    -lruntime -lascendcl -lstdc++ \
    -o "${build_dir}/color_inversion_bench"

"${build_dir}/color_inversion_bench"
