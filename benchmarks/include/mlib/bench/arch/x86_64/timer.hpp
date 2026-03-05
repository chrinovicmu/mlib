#pragma once 

#include <cstdint>
#include <ctime>
#include <immintrin.h>


namespace mlib::bench{

[[nodiscard]] inline uint64_t read_timer_start() noexcept {
    _mm_lfence();
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

[[nodiscard]] inline uint64_t read_timer_end() noexcept {
    uint32_t lo, hi, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    _mm_lfence();
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

[[nodiscard]] inline double timer_frequency_ghz() {
    struct timespec t0, t1, t_now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    uint64_t c0 = read_timer_start();
    do { clock_gettime(CLOCK_MONOTONIC_RAW, &t_now); }
    while ((t_now.tv_sec - t0.tv_sec) * 1'000'000'000LL +
           (t_now.tv_nsec - t0.tv_nsec) < 10'000'000LL);
    uint64_t c1 = read_timer_end();
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    int64_t ns = (t1.tv_sec  - t0.tv_sec)  * 1'000'000'000LL
               + (t1.tv_nsec - t0.tv_nsec);
    return static_cast<double>(c1 - c0) / static_cast<double>(ns);
}
inline double timer_ghz() {
    static double ghz = timer_frequency_ghz(); 
    return ghz;
}

inline double ticks_to_ns(uint64_t ticks) noexcept {
    return static_cast<double>(ticks) / timer_ghz();
}
}
