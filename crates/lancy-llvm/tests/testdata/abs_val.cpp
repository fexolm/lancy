// At -O1 clang lowers this to `llvm.abs.i64`, which the frontend
// doesn't understand yet — expected to skip.
extern "C" long abs_val(long x) { return x < 0 ? -x : x; }
