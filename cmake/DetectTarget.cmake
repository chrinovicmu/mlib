# =============================================================================
# cmake/DetectTarget.cmake
#
# Resolves MLIB_ARCH to exactly ONE concrete ISA string at configure
# time. That single string then drives three things, which therefore
# cannot disagree with one another:
#
#   1. which backend sources get compiled (.cpp, plus .S for assembly)
#   2. which MLIB_TARGET_ISA_<ISA> macro the generated config.hpp defines
#   3. which namespace ops.hpp aliases as kernels::active
#
# All three trace back to this variable, so "the build compiled AVX2 but
# the header dispatched to NEON" is not a reachable state.
#
# -----------------------------------------------------------------------------
# HOW MLIB_ARCH=auto IS RESOLVED  (three independent gates)
#
# A candidate ISA is selected only if ALL THREE hold. They are genuinely
# independent questions and checking fewer than three is a known bug class:
#
#   1. DOES THE CPU HAVE IT, AND DOES THE OS SAVE ITS REGISTERS?
#      Answered by actually running cmake/probe/cpu_probe.cpp on this
#      machine via try_run. Not inferrable from the compiler or the triple.
#
#   2. DOES THE COMPILER ACCEPT THE FLAGS?
#      Answered by check_cxx_source_compiles with the ISA's flag set.
#      A new-enough CPU with an old-enough compiler is a real combination.
#
#   3. HAS THE KERNEL BEEN WRITTEN?
#      Answered by EXISTS on every source for the ISA. During an incremental
#      port most backends do not exist yet.
#
# Gate 1 without gate 2 fails to build. Gate 2 without gate 1 builds
# perfectly and then SIGILLs. That second one is the trap, and it is worth
# spelling out because the obvious implementation walks straight into it:
#
#      check_cxx_compiler_flag("-mavx512f" HAS_AVX512)      # WRONG
#
# That tests the COMPILER. Every modern clang accepts it, on every host,
# including a 2013 Haswell with no AVX-512 silicon in it. Measured on the
# machine this was written on -- an Apple M2, which implements neither SVE
# nor SME -- Apple clang ACCEPTS both `-march=armv8-a+sve` and
# `-march=armv9-a+sme` without complaint. A flags-only check would have
# selected SVE here and produced a library that faults on its first kernel
# call. Gate 1 is what prevents that.
#
# -----------------------------------------------------------------------------
# WHY THE PROBE RETURNS A LADDER RATHER THAN ONE ANSWER
#
# cpu_probe.cpp prints every ISA the CPU supports, widest first, always
# ending in `generic` -- e.g. `avx512;avx2;avx;generic`. Gates 2 and 3 are
# then applied walking DOWN that list.
#
# If the probe returned only its best answer, a Zen 4 host would report
# avx512, find no avx512.cpp, and collapse all the way to `generic` --
# skipping the avx2 kernel sitting right there. The ladder makes "best
# AVAILABLE" mean what it says.
# =============================================================================

include(CheckCXXSourceCompiles)

# The user-facing knob. CACHE STRING so it persists across re-configures
# and shows up in ccmake / cmake-gui.
set(MLIB_ARCH "auto" CACHE STRING
    "Target ISA: auto|generic|avx|avx2|avx512|neon|sve|sme|rvv")

# Populates the drop-down in the GUI front-ends. Purely cosmetic -- it does
# NOT validate the value, which is why the explicit-request path below
# still validates against MLIB_KNOWN_ISAS by hand.
set_property(CACHE MLIB_ARCH PROPERTY STRINGS
    auto generic avx avx2 avx512 neon sve sme rvv)

# The authoritative list. Anything not in here is a typo, and is rejected
# rather than silently falling through to generic -- `-DMLIB_ARCH=avx3`
# should tell you avx3 is not a thing, not hand you a scalar build and let
# you discover it in the benchmark numbers three days later.
set(MLIB_KNOWN_ISAS generic avx avx2 avx512 neon sve sme rvv)

# -----------------------------------------------------------------------------
# Per-ISA compile flags.
#
# Declared BEFORE detection because detection consults them: gate 2 asks
# whether the compiler accepts a candidate's flags, so the table has to
# already exist when the ladder is walked.
#
# These are applied ONLY to that ISA's sources, never
# globally. A global -mavx2 would license the compiler to emit AVX2 into
# EVERY file in the project -- including any code that must run before an
# ISA check, and including the test harness. That is how a library ends up
# crashing in its own startup path on a machine it claims to support.
#
# Written as CMake LISTS, not strings. A quoted empty string passed where a
# list is expected survives as a literal empty argv element and reaches the
# compiler as a zero-length argument, which add_custom_command in
# particular handles badly. `set(X "")` produces an empty list; the
# distinction matters at the use site in CMakeLists.txt.
# -----------------------------------------------------------------------------
set(MLIB_ISA_FLAGS_generic "")                              # baseline, no flags
set(MLIB_ISA_FLAGS_avx     -mavx)                           # 256-bit, no FMA
set(MLIB_ISA_FLAGS_avx2    -mavx2 -mfma)                    # FMA is separate from AVX2
set(MLIB_ISA_FLAGS_avx512  -mavx512f -mavx512dq -mavx512bw) # F=core, DQ=doubleword/quadword, BW=byte/word
set(MLIB_ISA_FLAGS_neon    "")                              # mandatory on AArch64 -- no flag exists to enable it
set(MLIB_ISA_FLAGS_sve     -march=armv8-a+sve)
set(MLIB_ISA_FLAGS_sme     -march=armv9-a+sme)
set(MLIB_ISA_FLAGS_rvv     -march=rv64gcv)                  # g=IMAFD, c=compressed, v=vector

# -----------------------------------------------------------------------------
# ISA -> processor family, and thence to a source path.
#
# src/backends/ is laid out by FAMILY, not as one flat pile of .cpp files:
#
#     src/backends/simd_kernels.inc     shared algorithms for all SIMD ISAs
#     src/backends/generic.cpp          portable scalar, belongs to no family
#     src/backends/x86_64/              avx.cpp  avx2.cpp  avx512.cpp/.S
#     src/backends/arm/                 neon.cpp  (sve, sme -- Phase 6)
#     src/backends/riscv/               rvv.cpp/.S
#
# The grouping is not cosmetic. Backends within a family share an
# intrinsics header, a mental model, and usually a bug: if vfmaq_f32's
# addend-first argument order is wrong in neon.cpp it will be wrong in
# sve.cpp too, and having them adjacent is how that gets caught. Across
# families all SIMD backends share simd_kernels.inc. AVX-512 and RVV
# supply their trait primitives through external assembly functions.
#
# `generic` is deliberately NOT under a family directory. It is portable
# C++ with no intrinsics at all, it is compiled into every build regardless
# of the active ISA, and filing it under one processor family would imply
# a relationship it does not have.
# -----------------------------------------------------------------------------
set(MLIB_ISA_FAMILY_generic "")        # no family -- lives at the top level
set(MLIB_ISA_FAMILY_avx     x86_64)
set(MLIB_ISA_FAMILY_avx2    x86_64)
set(MLIB_ISA_FAMILY_avx512  x86_64)
set(MLIB_ISA_FAMILY_neon    arm)
set(MLIB_ISA_FAMILY_sve     arm)
set(MLIB_ISA_FAMILY_sme     arm)
set(MLIB_ISA_FAMILY_rvv     riscv)

# Absolute paths to an ISA's implementation files. Every place that needs to
# know where a backend lives goes through here, so the layout is described
# in exactly one spot and moving a family directory is a one-line edit.
function(mlib_isa_sources ISA OUT_VAR)
    set(_family "${MLIB_ISA_FAMILY_${ISA}}")
    if(_family STREQUAL "")
        set(_base "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/${ISA}")
    else()
        set(_base "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/${_family}/${ISA}")
    endif()
    set(_sources "${_base}.cpp")
    if(ISA STREQUAL "avx512" OR ISA STREQUAL "rvv")
        list(APPEND _sources "${_base}.S")
    endif()
    set(${OUT_VAR} "${_sources}" PARENT_SCOPE)
endfunction()

# Return the first missing source, or an empty string when all are present.
# An assembly backend is unavailable if either its traits or its .S is missing.
function(mlib_isa_missing_source ISA OUT_VAR)
    mlib_isa_sources(${ISA} _sources)
    set(${OUT_VAR} "" PARENT_SCOPE)
    foreach(_source IN LISTS _sources)
        if(NOT EXISTS "${_source}")
            set(${OUT_VAR} "${_source}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

# -----------------------------------------------------------------------------
# GATE 2 -- does this toolchain accept the ISA's flags?
#
# check_cxx_source_compiles rather than check_cxx_compiler_flag because our
# flag sets are multi-element (avx2 is `-mavx2 -mfma`, avx512 is three
# flags) and check_cxx_compiler_flag takes exactly one flag. Passing the
# whole set through CMAKE_REQUIRED_FLAGS tests them the way they will
# actually be used.
#
# CMAKE_REQUIRED_FLAGS is a space-separated STRING, not a list. Assigning a
# CMake list to it directly would embed literal semicolons in the command
# line, so it is joined first.
#
# CMAKE_REQUIRED_FLAGS is set as a plain local: CMake functions have their
# own variable scope, so this assignment cannot leak into the caller or
# into any later check.
# -----------------------------------------------------------------------------
function(mlib_isa_flags_supported ISA OUT_VAR)
    set(_flags ${MLIB_ISA_FLAGS_${ISA}})

    # An empty flag set is trivially supported. Short-circuiting also skips
    # a pointless compiler invocation for `generic` and `neon`, which is
    # the common case.
    if(NOT _flags)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()

    string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${_flags}")

    # The result variable name embeds the ISA. check_cxx_source_compiles
    # CACHES by that name and will not re-run if it is already set, so a
    # shared name across ISAs would make every ISA inherit the first one's
    # answer.
    # Assembly entry points use these ABIs. Reject an incompatible target
    # during selection, before the assembler or the linker is invoked.
    set(_probe "int main() { return 0; }")
    if(ISA STREQUAL "avx512")
        string(PREPEND _probe "
#if !defined(__x86_64__) || defined(_WIN32)
#error AVX-512 assembly requires x86-64 System V (ELF or Mach-O)
#endif
")
    elseif(ISA STREQUAL "rvv")
        string(PREPEND _probe "
#if !defined(__riscv) || __riscv_xlen != 64 || !defined(__riscv_vector)
#error RVV assembly requires RV64 with the V extension
#endif
#if !defined(__riscv_float_abi_double)
#error RVV assembly requires an LP64D toolchain
#endif
#include <riscv_vector.h>
using mlib_probe_rvv_f32 = vfloat32m1_t;
using mlib_probe_rvv_f64 = vfloat64m1_t;
")
    endif()
    check_cxx_source_compiles("${_probe}" MLIB_FLAGS_OK_${ISA})

    set(${OUT_VAR} ${MLIB_FLAGS_OK_${ISA}} PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# GATE 1 -- what does the CPU actually have?
#
# Compiles and RUNS cmake/probe/cpu_probe.cpp on this machine. Sets OUT_VAR
# to the ladder it printed, or to the empty string if anything went wrong,
# which the caller reads as "probe unusable, fall back".
#
# OUT_VAR is the name of a variable in the CALLER's scope. CMake functions
# get their own scope, so a plain set() here would be invisible outside --
# hence PARENT_SCOPE on every return path.
# -----------------------------------------------------------------------------
function(mlib_probe_host_ladder OUT_VAR)
    set(_probe_src "${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe/cpu_probe.cpp")

    if(NOT EXISTS "${_probe_src}")
        message(WARNING "mlib: ${_probe_src} is missing -- cannot probe CPU.")
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    # The three-argument (bindir) form of try_run. The scratch directory is
    # named so its leftovers are identifiable in CMakeFiles/ when a probe
    # failure needs diagnosing.
    #
    # CMAKE_CXX_STANDARD is forwarded explicitly rather than relying on
    # CMP0067 propagation, because a try_run that silently compiles as
    # C++98 would fail on nullptr and be reported as "no SIMD detected".
    try_run(_probe_run_rc _probe_compiled
            "${CMAKE_BINARY_DIR}/CMakeFiles/mlib_cpu_probe"
            "${_probe_src}"
            CMAKE_FLAGS "-DCMAKE_CXX_STANDARD=17"
            COMPILE_OUTPUT_VARIABLE _probe_compile_log
            RUN_OUTPUT_VARIABLE _probe_stdout)

    if(NOT _probe_compiled)
        message(WARNING
            "mlib: CPU probe failed to COMPILE. Falling back to conservative "
            "family inference; the build will be correct but may not use the "
            "widest available ISA.\n--- compiler output ---\n"
            "${_probe_compile_log}")
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    # try_run sets the run-result variable to the literal string
    # FAILED_TO_RUN when the binary built but could not be executed. That is
    # not a number, so it must be compared as a string BEFORE any numeric
    # comparison is attempted.
    if(_probe_run_rc STREQUAL "FAILED_TO_RUN")
        message(WARNING
            "mlib: CPU probe built but could not be executed. Falling back "
            "to conservative family inference.")
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    if(NOT _probe_run_rc EQUAL 0)
        message(WARNING
            "mlib: CPU probe exited ${_probe_run_rc}. Falling back to "
            "conservative family inference.")
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    # RUN_OUTPUT_VARIABLE captures stdout AND stderr merged. The probe is
    # written to keep stderr silent, but a linker note or sanitizer banner
    # landing in there would corrupt the ladder, so the shape of the result
    # is validated rather than trusted.
    string(STRIP "${_probe_stdout}" _ladder)

    if(NOT _ladder MATCHES "^[a-z0-9]+(;[a-z0-9]+)*$")
        message(WARNING
            "mlib: CPU probe produced unparseable output [${_ladder}]. "
            "Falling back to conservative family inference.")
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    set(${OUT_VAR} "${_ladder}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# Fallback ladder -- processor-FAMILY inference.
#
# Used only when the probe could not run. It answers the weaker question
# "what is architecturally guaranteed for this family", never "what does
# this chip have", so it deliberately claims as little as possible.
#
# Everything optional is omitted: AVX/AVX2/AVX-512 on x86-64, SVE/SME on
# AArch64, V on RISC-V. Guessing high here is precisely the SIGILL failure
# this file exists to prevent, and a fallback path is the worst place to
# take that risk since by definition nothing verified the guess.
# -----------------------------------------------------------------------------
function(mlib_fallback_ladder OUT_VAR)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        # Three spellings because the string comes from the OS, not from
        # CMake: Linux says x86_64, Windows says AMD64, some BSDs say amd64.
        #
        # SSE2 is the widest ISA guaranteed on every x86-64 part, and
        # `generic` compiles down to it anyway, so there is no rung above
        # generic that can be claimed without asking the CPU.
        set(${OUT_VAR} "generic" PARENT_SCOPE)

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        # NEON is architecturally MANDATORY on AArch64 -- part of the base
        # ISA, not an optional extension. This is the one case where family
        # inference is exactly as strong as a CPU probe: no AArch64 part
        # exists without it, so no runtime check could make the answer more
        # true.
        set(${OUT_VAR} "neon;generic" PARENT_SCOPE)

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv")
        # Both spellings appear in the wild. The V extension is OPTIONAL --
        # rv64gc parts have no vector unit at all -- so it is not claimed.
        set(${OUT_VAR} "generic" PARENT_SCOPE)

    else()
        # Unknown target. `generic` is portable C++ by construction, so this
        # is a working answer rather than an error.
        set(${OUT_VAR} "generic" PARENT_SCOPE)
    endif()
endfunction()

# =============================================================================
# Resolution
# =============================================================================

# Fat-binary mode is documented in the design but not built. Failing loudly
# with a pointer to the doc beats silently building something else and
# leaving the user to discover the difference at run time.
if(MLIB_ARCH STREQUAL "all")
    message(FATAL_ERROR
        "mlib: MLIB_ARCH=all (fat-binary mode) is documented in "
        "docs/fat-binary-mode.md but not implemented. Pick one ISA.")
endif()

if(MLIB_ARCH STREQUAL "auto")
    # -------------------------------------------------------------------------
    # AUTO: probe, then walk the ladder through gates 2 and 3.
    # -------------------------------------------------------------------------
    if(CMAKE_CROSSCOMPILING)
        # try_run cannot execute a target binary on the build host, and
        # inspecting the build host instead would answer a different
        # question than the one asked -- detecting NEON on an M2 while
        # building for riscv64 is not a degraded answer, it is a wrong one.
        #
        # Refuse rather than guess. A hard error at configure time is
        # recoverable in one command; a mis-targeted binary is not
        # diagnosable until it crashes on the target.
        message(FATAL_ERROR
            "mlib: MLIB_ARCH=auto is not valid when cross-compiling "
            "(target ${CMAKE_SYSTEM_PROCESSOR}). Pass -DMLIB_ARCH=<isa> "
            "explicitly.")
    endif()

    mlib_probe_host_ladder(_ladder)

    if(NOT _ladder)
        mlib_fallback_ladder(_ladder)
        message(STATUS "mlib: CPU probe unavailable; inferred ladder = ${_ladder}")
    else()
        message(STATUS "mlib: CPU probe ladder = ${_ladder}")
    endif()

    # The widest thing the SILICON can do, recorded before gates 2 and 3
    # narrow it. Reported in the configure summary so "detected avx512,
    # building generic" is visible rather than something you infer from a
    # disappointing benchmark.
    list(GET _ladder 0 MLIB_DETECTED_ISA)

    set(_chosen "")
    foreach(_cand IN LISTS _ladder)
        # ---- GATE 3: has the kernel been written? ----
        # This is what keeps the tree buildable during an incremental port.
        # Without it, adding `sme` to the ISA list would break every
        # configure until sme.cpp landed, forcing all eight backends to
        # arrive in one commit.
        mlib_isa_missing_source(${_cand} _missing_src)
        if(_missing_src)
            file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_missing_src}")
            message(STATUS "mlib:   ${_cand} -- no ${_rel} yet, skipping")
            continue()
        endif()

        # ---- GATE 2: does the compiler accept the flags? ----
        mlib_isa_flags_supported(${_cand} _flags_ok)
        if(NOT _flags_ok)
            message(STATUS "mlib:   ${_cand} -- compiler rejects flags or target ABI (${MLIB_ISA_FLAGS_${_cand}}), skipping")
            continue()
        endif()

        set(_chosen "${_cand}")
        break()
    endforeach()

    if(NOT _chosen)
        # Unreachable in practice: `generic` is the last rung of every
        # ladder, has no flags to reject, and src/backends/generic.cpp is
        # always present. Handled anyway so a future edit that breaks one
        # of those three assumptions produces a working build rather than
        # an empty MLIB_ARCH that fails somewhere less obvious.
        set(_chosen "generic")
    endif()

    # Overwrite the local (non-cache) MLIB_ARCH. The CACHE entry keeps
    # saying "auto", which is intentional: re-running cmake re-probes
    # rather than freezing whatever the first configure happened to find --
    # so installing a newer compiler, or writing neon.cpp, takes effect on
    # the next configure without needing the cache blown away.
    set(MLIB_ARCH "${_chosen}")

else()
    # -------------------------------------------------------------------------
    # EXPLICIT: the user named an ISA. Honour it, or say precisely why not.
    #
    # Gate 1 is deliberately NOT applied here. Building AVX-512 kernels on a
    # machine that cannot execute them is a legitimate thing to want --
    # cross-builds, CI matrices, inspecting generated assembly -- and
    # refusing it would make the library untestable from one host.
    # -------------------------------------------------------------------------
    # Captured BEFORE the gate-3 fallback below, which may rewrite
    # MLIB_ARCH to generic. Reading it afterwards would report "generic
    # (explicit, not probed)" for a build the user explicitly asked to be
    # neon -- which hides exactly the substitution the summary exists to
    # surface.
    set(MLIB_DETECTED_ISA "${MLIB_ARCH} (explicit, not probed)")

    # list(JOIN) rather than bare ${MLIB_KNOWN_ISAS}: a CMake list
    # interpolated inside a quoted string keeps its literal separators, so
    # the unjoined form prints "generic;avx;avx2;..." at the user.
    list(JOIN MLIB_KNOWN_ISAS " " _known_pretty)

    if(NOT MLIB_ARCH IN_LIST MLIB_KNOWN_ISAS)
        message(FATAL_ERROR
            "mlib: MLIB_ARCH=${MLIB_ARCH} is not a known ISA. "
            "Valid values: auto all ${_known_pretty}")
    endif()

    # Gate 2 is FATAL rather than a fallback on this path. If the toolchain
    # cannot emit the requested ISA there is nothing this build can do about
    # it, and quietly producing a scalar library while the user believes
    # they are benchmarking AVX-512 is worse than not building at all.
    mlib_isa_flags_supported(${MLIB_ARCH} _flags_ok)
    if(NOT _flags_ok)
        message(FATAL_ERROR
            "mlib: MLIB_ARCH=${MLIB_ARCH} was requested, but this compiler "
            "rejects its flags or target ABI (${MLIB_ISA_FLAGS_${MLIB_ARCH}}) for target "
            "${CMAKE_SYSTEM_PROCESSOR}. Use a cross-toolchain or pick "
            "another ISA.")
    endif()

    # Gate 3 stays a WARNING plus fallback, not an error, for the same
    # incremental-port reason as above -- the kernel is missing because we
    # have not written it yet, which is our problem and a temporary one.
    # WARNING rather than STATUS because silently building something other
    # than what was asked for is exactly the kind of thing that wastes an
    # afternoon of benchmarking.
    mlib_isa_missing_source(${MLIB_ARCH} _missing_src)
    if(_missing_src)
        file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_missing_src}")
        message(WARNING
            "mlib: ${_rel} does not exist yet -- "
            "falling back to MLIB_ARCH=generic.")
        set(MLIB_ARCH "generic")
    endif()

endif()

# Uppercased form, consumed by config.hpp.in as
# `#define MLIB_TARGET_ISA_@MLIB_ARCH_UPPER@ 1`.
string(TOUPPER "${MLIB_ARCH}" MLIB_ARCH_UPPER)

# Indirect variable lookup: expands MLIB_ARCH first, then reads the
# variable whose name that spells. With MLIB_ARCH=avx2 this reads
# MLIB_ISA_FLAGS_avx2.
set(MLIB_ACTIVE_ISA_FLAGS ${MLIB_ISA_FLAGS_${MLIB_ARCH}})

# Resolve all sources, including an assembly companion when needed.
mlib_isa_sources(${MLIB_ARCH} MLIB_ACTIVE_ISA_SOURCES)
