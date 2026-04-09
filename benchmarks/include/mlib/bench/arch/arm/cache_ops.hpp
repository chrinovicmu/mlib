#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace mlib::bench {

inline void memory_fence() noexcept {
    __asm__ volatile("dmb ish" ::: "memory");
}

inline void flush_cache_line(const void* ptr) noexcept {
#if defined(__linux__) && !defined(__APPLE__)
    __asm__ volatile(
        "dc civac, %0"
        :
        : "r"(ptr)
        : "memory"
    );
#else
    (void)ptr;
#endif
    memory_fence();
}

inline void flush_buffer(const void* ptr, size_t bytes) noexcept {

    constexpr size_t cache_line = 64;
    const char* p = static_cast<const char*>(ptr);

    for (size_t off = 0; off < bytes; off += cache_line) {
        flush_cache_line(p + off);
    }
    memory_fence();
}

inline void warm_buffer(const void* ptr, size_t bytes) noexcept {
   
    constexpr size_t cache_line = 64;
    volatile const char* p = static_cast<volatile const char*>(ptr);
    volatile char sink = 0;

    for (size_t off = 0; off < bytes; off += cache_line) {
        sink ^= p[off];
    }
    (void)sink;
    __asm__ volatile("isb" ::: "memory");
}

} // namespace mlib::bench
