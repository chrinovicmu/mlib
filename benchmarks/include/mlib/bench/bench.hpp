#pragma once

#include "stats.hpp"
#include "perf_counters.hpp"


#if defined(MLIB_ARCH_X86_64)
#   include "arch/x86_64/timer.hpp"
#   include "arch/x86_64/cache_ops.hpp"
#elif defined(MLIB_ARCH_ARM)
#   include "arch/arm/timer.hpp"
#   include "arch/arm/cache_ops.hpp"
#else
#   error "Unknown architecture. Define MLIB_ARCH_X86_64 or MLIB_ARCH_ARM in CMake."
#endif

#include "arch/detect.hpp"
#include "runner.hpp"
