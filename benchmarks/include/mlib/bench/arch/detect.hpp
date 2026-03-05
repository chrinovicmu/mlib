#pragma once 

#include <string>
#include <cstdint>
#include <array>

#if defined(__x86_64__) || defined(__i386__)
#  include <cpuid.h>
#endif

namespace mlib::bench{

struct CpuInfo{

    std::string brand;
    std::string microarch; 
    std::string arch; 
    uint32_t family = 0; 
    uint32_t model = 0; 
    uint32_t stepping = 0; 
}

#if defined(__x86_64__) || defined(__i386__)

inline std::string intel_microarch(uint32_t family, uint32_t model) {

    if (family == 6) {
        switch (model) {
        // Skylake (client)
        case 0x4E: case 0x5E: return "Skylake";
        // Skylake-X (server/HEDT)
        case 0x55: return "Skylake-X / Cascade Lake";
        // Kaby Lake
        case 0x8E: case 0x9E: return "Kaby Lake / Coffee Lake";
        // Ice Lake (client)
        case 0x7E: return "Ice Lake";
        // Tiger Lake
        case 0x8C: case 0x8D: return "Tiger Lake";
        // Alder Lake (hybrid P+E)
        case 0x97: case 0x9A: return "Alder Lake";
        // Raptor Lake
        case 0xB7: case 0xBA: return "Raptor Lake";
        // Zen (AMD also reports family 6 on some models via CPUID compat)
        default: return "Intel family6/model=" + std::to_string(model);
        }
    }
    if (family == 0x19) return "Zen 3 / Zen 4 (AMD)";
    if (family == 0x17) return "Zen / Zen+ / Zen 2 (AMD)";
    return "Unknown x86 family=" + std::to_string(family);
}

inline CpuInfo detect_cpu() {
    CpuInfo info;
    info.arch = "x86_64";

    std::array<uint32_t, 12> brand_raw{};
    for (int i = 0; i < 3; ++i) {
        __cpuid(0x80000002 + i,
                brand_raw[i*4+0], brand_raw[i*4+1],
                brand_raw[i*4+2], brand_raw[i*4+3]);
    }
    info.brand = std::string(reinterpret_cast<const char*>(brand_raw.data()), 48);
    
    while (!info.brand.empty() &&
           (info.brand.back() == '\0' || info.brand.back() == ' '))
        info.brand.pop_back();

    uint32_t eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);

    // EAX bit layout:
    //   [3:0]   stepping
    //   [7:4]   base model
    //   [11:8]  base family
    //   [19:16] extended model   → DisplayModel  = (extended<<4) | base  (if family==6)
    //   [27:20] extended family  → DisplayFamily = base + extended       (if base==0xF)
    uint32_t base_family   = (eax >>  8) & 0xF;
    uint32_t base_model    = (eax >>  4) & 0xF;
    uint32_t ext_family    = (eax >> 20) & 0xFF;
    uint32_t ext_model     = (eax >> 16) & 0xF;

    info.stepping = (eax >>  0) & 0xF;
    info.family   = (base_family == 0xF) ? (base_family + ext_family) : base_family;
    info.model    = (base_family == 6 || base_family == 0xF)
                      ? ((ext_model << 4) | base_model)
                      : base_model;

    info.microarch = intel_microarch(info.family, info.model);
    return info;
}

#elif defined(__aarch64__)

inline CpuInfo detect_cpu() {
    CpuInfo info;
    info.arch = "aarch64";

    uint64_t midr;
    __asm__ volatile("mrs %0, midr_el1" : "=r"(midr));

    uint32_t implementer = (midr >> 24) & 0xFF;
    uint32_t part_num    = (midr >>  4) & 0xFFF;

    if (implementer == 0x41) { // ARM Ltd
        switch (part_num) {
        case 0xD05: info.microarch = "ARM Cortex-A55";  break;
        case 0xD0B: info.microarch = "ARM Cortex-A76";  break;
        case 0xD0C: info.microarch = "ARM Neoverse-N1"; break;
        case 0xD40: info.microarch = "ARM Neoverse-V1"; break;
        default:    info.microarch = "ARM part=0x" + std::to_string(part_num);
        }
    } else if (implementer == 0x61) { // Apple
        switch (part_num) {
        case 0x022: info.microarch = "Apple M1 (Icestorm/Firestorm)"; break;
        case 0x023: info.microarch = "Apple M2 (Blizzard/Avalanche)"; break;
        case 0x024: info.microarch = "Apple M3"; break;
        default:    info.microarch = "Apple Silicon part=0x" + std::to_string(part_num);
        }
    } else {
        info.microarch = "AArch64 implementer=0x" + std::to_string(implementer);
    }

    info.brand = info.microarch;
    return info;
}

#else
inline CpuInfo detect_cpu() {
    return { "Unknown", "Unknown", "Unknown", 0, 0, 0 };
}
#endif

inline const CpuInfo& cpu_info() {
    static CpuInfo info = detect_cpu();
    return info;
}

inline std::string results_dir_name() {
    const auto& ci = cpu_info();
    std::string dir = ci.microarch;
    for (char& c : dir) {
        if (c == ' ' || c == '/') c = '_';
        if (c >= 'A' && c <= 'Z') c += 32;
    }
    return dir;
}
}

