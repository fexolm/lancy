// Allocates a std::string of N 'a' characters and returns its size.
// SBO-or-heap, construct/destruct — expected to skip.
#include <string>

extern "C" long string_build(long n) {
    std::string s(static_cast<std::size_t>(n), 'a');
    return static_cast<long>(s.size());
}
