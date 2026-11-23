#include <cstdio>

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#include <cpuid.h>

static unsigned long long read_xcr0() {
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0"
                         : "=a"(eax), "=d"(edx)
                         : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
}

static void emit_ladder() {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    const bool osxsave = (ecx & (1u << 27)) != 0;

    // ECX bit 28 -- AVX silicon present.
    const bool cpu_avx = (ecx & (1u << 28)) != 0;

    const bool cpu_fma = (ecx & (1u << 12)) != 0;

    unsigned int max_leaf = __get_cpuid_max(0, nullptr);

    bool cpu_avx2 = false;
    bool cpu_avx512f = false;
    bool cpu_avx512dq = false;
    bool cpu_avx512bw = false;

    if (max_leaf >= 7) {
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        cpu_avx2     = (ebx & (1u << 5))  != 0;  // EBX[5]
        cpu_avx512f  = (ebx & (1u << 16)) != 0;  // EBX[16] foundation
        cpu_avx512dq = (ebx & (1u << 17)) != 0;  // EBX[17] doubleword/quadword
        cpu_avx512bw = (ebx & (1u << 30)) != 0;  // EBX[30] byte/word
    }
    unsigned long long xcr0 = osxsave ? read_xcr0() : 0ull;

    const bool os_ymm = (xcr0 & 0x6ull) == 0x6ull;

    const bool os_zmm = (xcr0 & 0xE6ull) == 0xE6ull;

    if (cpu_avx512f && cpu_avx512dq && cpu_avx512bw && os_zmm) {
        std::printf("avx512;");
    }
    if (cpu_avx2 && cpu_fma && os_ymm) {
        std::printf("avx2;");
    }
    if (cpu_avx && os_ymm) {
        std::printf("avx;");
    }
    std::printf("generic");
}

#elif defined(__aarch64__) || defined(_M_ARM64)

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

#if defined(__APPLE__)
static bool darwin_has(const char* name) {
    int value = 0;
    size_t len = sizeof(value);
    if (sysctlbyname(name, &value, &len, nullptr, 0) != 0) {
        return false;
    }
    return value != 0;
}
#endif

static void emit_ladder() {
    bool cpu_sve = false;

#if defined(__APPLE__)
    cpu_sve = darwin_has("hw.optional.arm.FEAT_SVE");
#elif defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    cpu_sve = (hwcap & (1ul << 22)) != 0;
#endif

    if (cpu_sve) {
        std::printf("sve;");
    }
    std::printf("neon;");
    std::printf("generic");

}

#elif defined(__riscv)

#if defined(__linux__)
#include <sys/auxv.h>
#endif

static void emit_ladder() {
    bool cpu_v = false;

#if defined(__linux__)
    unsigned long hwcap = getauxval(AT_HWCAP);
    cpu_v = (hwcap & (1ul << ('V' - 'A'))) != 0;
#endif

    if (cpu_v) {
        std::printf("rvv;");
    }
    std::printf("generic");
}

#else

static void emit_ladder() {
    std::printf("generic");
}

#endif

int main() {
    emit_ladder();
    return 0;
}
