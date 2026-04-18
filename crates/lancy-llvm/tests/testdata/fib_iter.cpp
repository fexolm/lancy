// Iterative Fibonacci. Loop-carried state forces a `phi` in the loop
// header — expected to skip until SSA destruction lands.
extern "C" long fib_iter(long n) {
    long a = 0, b = 1;
    for (long i = 0; i < n; ++i) {
        long c = a + b;
        a = b;
        b = c;
    }
    return a;
}
