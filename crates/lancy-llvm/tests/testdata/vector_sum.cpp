// std::vector + std::accumulate. Heap allocations, function calls
// (`new`, destructor, accumulate instantiation) — far beyond the
// current frontend. Expected to skip.
#include <numeric>
#include <vector>

extern "C" long vector_sum(long n) {
    std::vector<long> v;
    v.reserve(static_cast<std::size_t>(n));
    for (long i = 1; i <= n; ++i) v.push_back(i);
    return std::accumulate(v.begin(), v.end(), 0L);
}
