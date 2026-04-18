// Evaluate 3x^4 + 5x^3 - 2x^2 + 7x - 1 via Horner's method.
extern "C" long horner(long x) {
    long p = 3;
    p = p * x + 5;
    p = p * x - 2;
    p = p * x + 7;
    p = p * x - 1;
    return p;
}
