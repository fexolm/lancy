// Builds a map { i -> i*i } for i in [0,10) and looks up `key`.
// Red-black tree, heap allocator, comparator instantiation — expected
// to skip.
#include <map>

extern "C" long map_square_lookup(long key) {
    std::map<long, long> m;
    for (long i = 0; i < 10; ++i) m[i] = i * i;
    auto it = m.find(key);
    return it == m.end() ? -1 : it->second;
}
