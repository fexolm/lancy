// Iterative factorial — loop with phi in the header. Expected to skip.
extern "C" long factorial_iter(long n) {
    long r = 1;
    for (long i = 2; i <= n; ++i) r *= i;
    return r;
}
