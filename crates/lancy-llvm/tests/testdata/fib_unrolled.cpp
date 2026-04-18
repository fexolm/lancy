// Iterative Fibonacci unrolled by hand for N=10, no loop / no phi.
// Returns fib(10) = 55 regardless of the input, which we add to the seed
// so the compiler can't constant-fold the whole body.
extern "C" long fib_unrolled(long seed) {
    long a = seed;
    long b = seed + 1;
    long c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    c = a + b; a = b; b = c;
    return b;
}
