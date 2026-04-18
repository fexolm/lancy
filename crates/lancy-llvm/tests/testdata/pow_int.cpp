// Integer exponentiation by squaring. Loop + `and`/`shr` — expected to
// skip.
extern "C" long pow_int(long base, long exp) {
    long r = 1;
    while (exp > 0) {
        if (exp & 1) r *= base;
        base *= base;
        exp >>= 1;
    }
    return r;
}
