// tests/main.cpp -- runner shared by every test binary.

#include "harness.hpp"
#include <mlib/mlib.hpp>

#include <cstdio>

int main() {
    std::printf("mlib %d.%d.%d  |  ISA: %s\n", MLIB_VERSION_MAJOR,
                MLIB_VERSION_MINOR, MLIB_VERSION_PATCH,
                MLIB_TARGET_ISA_NAME);
    std::printf("running %zu tests\n\n", th::registry().size());

    int failed_tests = 0;
    for (const auto& tc : th::registry()) {
        const int before = th::failures();
        tc.fn();
        const int added = th::failures() - before;
        if (added) {
            std::printf("[FAIL] %s (%d check%s)\n", tc.name, added,
                        added == 1 ? "" : "s");
            ++failed_tests;
        } else {
            std::printf("[ ok ] %s\n", tc.name);
        }
    }

    std::printf("\n%d/%zu tests passed", 
                static_cast<int>(th::registry().size()) - failed_tests,
                th::registry().size());
    if (failed_tests) {
        std::printf("  --  %d FAILED, %d checks\n", failed_tests, th::failures());
        return 1;
    }
    std::printf("\n");
    return 0;
}
