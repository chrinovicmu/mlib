#!/bin/sh
# Requires CMake, Ninja, a GNU riscv64-linux-gnu C++ toolchain, and qemu-riscv64
# with RVV 1.0 support. The same library is tested at four vector lengths.
set -eu
mlib_source_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mlib_build_dir=${1:-"$mlib_source_dir/build-rvv"}
mlib_sysroot=${MLIB_RVV_SYSROOT:-/usr/riscv64-linux-gnu}

cmake -S "$mlib_source_dir" -B "$mlib_build_dir" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$mlib_source_dir/cmake/toolchains/riscv64-linux-gnu.cmake" \
    -DMLIB_ARCH=rvv -DMLIB_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CROSSCOMPILING_EMULATOR=qemu-riscv64;-L;$mlib_sysroot"
cmake --build "$mlib_build_dir" --parallel
for mlib_vlen in 128 256 512 1024; do
    printf 'Testing RVV with VLEN=%s bits\n' "$mlib_vlen"
    QEMU_CPU="rv64,v=true,vlen=$mlib_vlen,elen=64,vext_spec=v1.0" \
        ctest --test-dir "$mlib_build_dir" --output-on-failure
done
