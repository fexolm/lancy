// Long chain of dependent arithmetic — exercises register pressure and
// spill heuristics in the lancy allocator.
extern "C" long stress_arith(long a, long b) {
    long t1 = a + b;
    long t2 = a - b;
    long t3 = a * b;
    long t4 = t1 * t2;
    long t5 = t3 + t4;
    long t6 = t5 * a;
    long t7 = t6 + t5;
    long t8 = t7 - t1;
    long t9 = t8 * t2;
    long t10 = t9 + t3;
    long t11 = t10 * t4;
    long t12 = t11 - t6;
    return t12 + t10;
}
