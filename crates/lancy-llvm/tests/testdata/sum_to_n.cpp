// Sum of 1..=n. Clang at -O1 recognises the arithmetic series and emits
// `n*(n+1)/2` — ends up using `sdiv`/`lshr`. Expected to skip.
extern "C" long sum_to_n(long n) {
    long s = 0;
    for (long i = 1; i <= n; ++i) s += i;
    return s;
}
