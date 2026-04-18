// Euclidean GCD — loop + `srem`. Expected to skip on both counts.
extern "C" long gcd_euclid(long a, long b) {
    while (b != 0) {
        long t = b;
        b = a % b;
        a = t;
    }
    return a;
}
