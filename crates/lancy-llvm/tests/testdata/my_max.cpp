// At -O1 clang emits `llvm.smax.i64` — expected to skip.
extern "C" long my_max(long a, long b) { return a > b ? a : b; }
