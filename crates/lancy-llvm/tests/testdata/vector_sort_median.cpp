// Picks the median of three via std::sort. STL instantiation + heap.
// Expected to skip.
#include <algorithm>
#include <vector>

extern "C" long vector_sort_median(long a, long b, long c) {
    std::vector<long> v{a, b, c};
    std::sort(v.begin(), v.end());
    return v[1];
}
