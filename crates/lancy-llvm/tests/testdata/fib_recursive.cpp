// Recursive Fibonacci — self-calls, which the frontend / JIT doesn't
// resolve yet. Expected to skip.
extern "C" long fib_recursive(long n) {
    if (n < 2) return n;
    return fib_recursive(n - 1) + fib_recursive(n - 2);
}
