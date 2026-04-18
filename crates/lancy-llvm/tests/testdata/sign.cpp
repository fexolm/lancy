// At -O1 clang computes sign as `(x >> 63) | ((x | -x) >> 63)` style
// via `ashr` + `select` — expected to skip.
extern "C" long sign(long x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
