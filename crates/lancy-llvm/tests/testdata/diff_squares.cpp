// a^2 - b^2 as (a+b)*(a-b) to force two distinct sub-expressions.
extern "C" long diff_squares(long a, long b) {
    return (a + b) * (a - b);
}
