// (x + y)^3 expanded to individual mul/add — no intrinsic, no branches.
extern "C" long cube_sum(long x, long y) {
    return x*x*x + 3*x*x*y + 3*x*y*y + y*y*y;
}
