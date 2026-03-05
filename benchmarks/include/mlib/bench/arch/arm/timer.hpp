#pragma once

#include <cstdint>
#include <ctime>

namespace mlib::bench {

[[nodiscard]] inline uint64_t read_timer_start() noexcept {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

[[nodiscard]] inline uint64_t read_timer_end() noexcept {
    uint64_t val;
    __asm__ volatile("dsb nsh" ::: "memory");
    __asm__ volatile("isb"     ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

[[nodiscard]] inline double timer_frequency_ghz() {
    uint64_t freq_hz;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq_hz));
    return static_cast<double>(freq_hz) / 1e9;
}

inline double timer_ghz() {
    static double ghz = timer_frequency_ghz();
    return ghz;
}

inline double ticks_to_ns(uint64_t ticks) noexcept {
    return static_cast<double>(ticks) / timer_ghz();
}

} // namespace mlib::bench
