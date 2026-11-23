# Linux RV64/LP64D cross-build with the GNU toolchain.
# Select -DMLIB_ARCH=rvv separately; other sources stay at the RV64GC baseline.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gc -mabi=lp64d")
set(CMAKE_ASM_FLAGS_INIT "-march=rv64gc -mabi=lp64d")

# Optional at configure time:
# -DCMAKE_CROSSCOMPILING_EMULATOR="qemu-riscv64;-L;/usr/riscv64-linux-gnu;-cpu;rv64,v=true,vlen=256,elen=64"
# CTest then runs each test target through that emulator.
