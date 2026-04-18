// Uses all six SysV AMD64 integer-argument registers at the ABI boundary.
extern "C" long six_args(long a, long b, long c, long d, long e, long f) {
    return (a + b) * (c - d) + e * f - a;
}
