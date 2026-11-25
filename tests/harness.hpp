// tests/harness.hpp
//
// Minimal dependency-free test harness.
//

#pragma once

#include <cmath>    
#include <cstdio>   
#include <cstdlib>
#include <string>
#include <vector>

namespace th {

// capture, and this keeps the type trivially copyable.
struct TestCase {
    const char* name;
    void (*fn)();
};

// The registry, as a function-local static rather than a namespace-scope
// global.
//
// This is the standard fix for the static initialization order fiasco.
// Registrar objects (below) are constructed during static init in an
// order the standard does not define ACROSS translation units. If the
// registry were a namespace-scope global, a Registrar in one TU could run
// before the vector itself was constructed, and push_back onto raw
// memory. A function-local static is guaranteed to be constructed on
// first call, so the first Registrar to run creates it.
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

// Running count of failed CHECKS (not failed tests). Same
// function-local-static reasoning as registry(). The runner samples this
// before and after each test to attribute failures to the right test.
inline int& failures() {
    static int f = 0;
    return f;
}

// Registration happens as a side effect of constructing one of these at
// namespace scope -- see the TH_TEST macro.
struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

// Relative-error comparison with an absolute floor.
//
template <typename T>
inline bool close(T got, T want, double tol) {
    const double g = static_cast<double>(got);
    const double w = static_cast<double>(want);

    // NaN == NaN is false by IEEE rule, so an expected-NaN test could
    // never pass without this. Two NaNs are treated as matching
    // regardless of payload or signalling status.
    if (std::isnan(g) && std::isnan(w)) return true;

    // Infinities compare exactly: +inf matches only +inf. Falling through
    // to the relative test below would compute inf - inf = NaN and give a
    // meaningless answer.
    if (std::isinf(g) || std::isinf(w)) return g == w;

    const double diff = std::abs(g - w);
    const double mag  = std::abs(w);

    // The `mag > 1.0 ? mag : 1.0` clamp is what makes this both a
    // relative AND an absolute test:
    //   * for |want| > 1, the bound scales with the magnitude (relative)
    //   * for |want| <= 1, the bound is just tol (absolute)
    // Without the floor, an expected value of exactly 0.0 would demand
    // diff <= 0 -- bit equality -- which defeats the entire purpose above.
    return diff <= tol * (mag > 1.0 ? mag : 1.0);
}

}  // namespace th
// TH_TEST(name) -- declare and register a test in one statement.
//
#define TH_TEST(name)                                          \
    static void name();                                        \
    static ::th::Registrar th_reg_##name(#name, name);         \
    static void name()

// TH_CHECK(cond) -- assert a boolean, report and count on failure.
#define TH_CHECK(cond)                                                        \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "    FAIL %s:%d: %s\n", __FILE__, __LINE__,  \
                         #cond);                                              \
            ++::th::failures();                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// TH_CHECK_CLOSE(got, want, tol) -- float comparison via th::close.
//
#define TH_CHECK_CLOSE(got, want, tol)                                        \
    do {                                                                      \
        auto _g = (got);                                                      \
        auto _w = (want);                                                     \
        if (!::th::close(_g, _w, (tol))) {                                    \
            std::fprintf(stderr,                                              \
                         "    FAIL %s:%d: %s\n      got  %.17g\n"             \
                         "      want %.17g\n",                                \
                         __FILE__, __LINE__, #got,                            \
                         static_cast<double>(_g), static_cast<double>(_w));   \
            ++::th::failures();                                               \
        }                                                                     \
    } while (0)
// %.17g prints a double with enough significant digits to round-trip
// exactly, so a near-miss is distinguishable from an exact match in the
// output rather than both printing as "1.5".

// TH_CHECK_THROWS(expr, extype) -- assert that expr throws extype.
#define TH_CHECK_THROWS(expr, extype)                                         \
    do {                                                                      \
        bool _caught = false;                                                 \
        try {                                                                 \
            (void)(expr);                                                     \
        } catch (const extype&) {                                             \
            _caught = true;                                                   \
        } catch (...) {                                                       \
        }                                                                     \
        if (!_caught) {                                                       \
            std::fprintf(stderr, "    FAIL %s:%d: expected %s from %s\n",     \
                         __FILE__, __LINE__, #extype, #expr);                 \
            ++::th::failures();                                               \
        }                                                                     \
    } while (0)
