// Fused multiply-add — primitive building block for dot products.
extern "C" long mad(long a, long b, long c) { return a * b + c; }
