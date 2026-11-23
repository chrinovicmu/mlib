include(CheckCXXSourceCompiles)

set_property(CACHE MLIB_ARCH PROPERTY STRINGS
    auto generic avx avx2 avx512 neon sve sme rvv)


set(MLIB_KNOWN_ISAS generic avx avx2 avx512 neon sve sme rvv)

set(MLIB_ISA_FLAGS_generic "")                              # baseline, no flags
set(MLIB_ISA_FLAGS_avx     -mavx)                           # 256-bit, no FMA
set(MLIB_ISA_FLAGS_avx2    -mavx2 -mfma)                    # FMA is separate from AVX2
set(MLIB_ISA_FLAGS_avx512  -mavx512f -mavx512dq -mavx512bw) # F=core, DQ=doubleword/quadword, BW=byte/word
set(MLIB_ISA_FLAGS_neon    "")                              # mandatory on AArch64 -- no flag exists to enable it
set(MLIB_ISA_FLAGS_sve     -march=armv8-a+sve)
set(MLIB_ISA_FLAGS_sme     -march=armv9-a+sme)
set(MLIB_ISA_FLAGS_rvv     -march=rv64gcv)                  # g=IMAFD, c=compressed, v=vector


set(MLIB_ISA_FAMILY_generic "")        # no family -- lives at the top level
set(MLIB_ISA_FAMILY_avx     x86_64)
set(MLIB_ISA_FAMILY_avx2    x86_64)
set(MLIB_ISA_FAMILY_avx512  x86_64)
set(MLIB_ISA_FAMILY_neon    arm)
set(MLIB_ISA_FAMILY_sve     arm)
set(MLIB_ISA_FAMILY_sme     arm)
set(MLIB_ISA_FAMILY_rvv     riscv)

function(mlib_isa_source ISA OUT_VAR)
    set(_family "${MLIB_ISA_FAMILY_${ISA}}")
    if(_family STREQUAL "")
        set(${OUT_VAR} "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/${ISA}.cpp"
            PARENT_SCOPE)
    else()
        set(${OUT_VAR}
            "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/${_family}/${ISA}.cpp"
            PARENT_SCOPE)
    endif()
endfunction()

function(mlib_isa_flags_supported ISA OUT_VAR)
    set(_flags ${MLIB_ISA_FLAGS_${ISA}})

    if(NOT _flags)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()

    string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${_flags}")

    check_cxx_source_compiles("int main() { return 0; }" MLIB_FLAGS_OK_${ISA})

    set(${OUT_VAR} ${MLIB_FLAGS_OK_${ISA}} PARENT_SCOPE)
endfunction()


function(mlib_probe_host_ladder OUT_VAR)
    set(_probe_src "${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe/cpu_probe.cpp")

    if(NOT EXISTS "${_probe_src}")
        message(WARNING "mlib: ${_probe_src} is missing -- cannot probe CPU.")
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()
    
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

function(mlib_fallback_ladder OUT_VAR)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(${OUT_VAR} "generic" PARENT_SCOPE)

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(${OUT_VAR} "neon;generic" PARENT_SCOPE)

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv")
        set(${OUT_VAR} "generic" PARENT_SCOPE)

    else()
        set(${OUT_VAR} "generic" PARENT_SCOPE)
    endif()
endfunction()



