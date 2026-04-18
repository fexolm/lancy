// Lowers to nested `select` at -O1 — expected to skip.
extern "C" long clamp(long x, long lo, long hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
