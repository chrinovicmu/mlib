#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
 
#include <emmintrin.h>
#include <immintrin.h>   
#include <cpuid.h>       
#include <xmmintrin.h>

namespace  mlib{
namespace  bench{
    
namespace detail{

inline bool detect_clflushopt() noexcept {
    uint32_t eax, ebx, ecx, edx;
 
    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 7) return false;
 
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx >> 23) & 1u;
}

inline bool has_clflushopt() noexcept {
    static bool val = detect_clflushopt();
    return val;
}

}// namespace detail 

inline void_memory_fence() noexcept{
    _mm_mfence(); 
}

inline void flush_cache_line(const void* ptr) noexcept{

    if(detail::has_clflushopt()){
        _mm_clflushopt(ptr); 
    }else{
        _mm_clflush(ptr); 
    }
}

inline void flush_buffer(const void* ptr, size_t bytes) noexcept{

    constexpr size_t cache_line_size = 64; 

    /*align  starting address down to the nearest cacheline boundary*/ 
    const uintptr_t start_addr = reinterpret_cast<uintptr_t>(ptr) 
        & ~static_cast<uintptr_t>(63); 

    const uintptr_t end_addr = reinterpret_cast<uintptr_t>(ptr) + bytes; 
    const char *p = reinterpret_cast<const char*>(start_addr); 

    const char* const end_p = reinterpret_cast<const char*>(end_addr); 
    const char* const unroll_end = end_p - (8 * cache_line_size - 1); 
    
    while (p < unroll_end) {
        // Issue 8 flush instructions back-to-back with no branching between them.
        // The CPU's memory subsystem can buffer multiple in-flight flushes
        // simultaneously, so this keeps the flush queue saturated.
        flush_cache_line(p);
        flush_cache_line(p + 1 * cache_line_size);
        flush_cache_line(p + 2 * cache_line_size);
        flush_cache_line(p + 3 * cache_line_size);
        flush_cache_line(p + 4 * cache_line_size);
        flush_cache_line(p + 5 * cache_line_size);
        flush_cache_line(p + 6 * cache_line_size);
        flush_cache_line(p + 7 * cache_line_size);
        p += 8 * cache_line_size;
    }
 
    while (p < end_p) {
        flush_cache_line(p);
        p += cache_line_size;
    }
 
    memory_fence();
}

inline void warm_buffer(const void *ptr, size_t bytes) noexcept{

    constexpr size_t cache_line_size = 64; 
    constexpr size_t prefetch_distance = 512; 

    const char *p = static_cast<const char*>(ptr); 
    const char* end = p + bytes; 

    for(const char* q = p; q < end; q += cache_line_size){
        const char* prefetch_target = q + prefetch_distance; 
        
        if(prefetch_target < end){
            _mm_prefetch(prefetch_target, _MM_HINT_T0); 

        }
    }

    volatile const char* vp = static_cast<volatile const char*>(ptr);
    volatile char sink = 0;  // volatile: every write is a real store, preventing
                              // the compiler from treating this as dead code
 
    for (size_t offset = 0; offset < bytes; offset += cache_line_size) {
        sink ^= vp[offset];
    }
    (void)sink;
    _mm_lfence();
}

}// namespace bench    
}// namespace mlib 
