/*
Use the Thrust library to find the sum of a large array of integers on the GPU. Include the Thrust headers, create thrust::device_vector wrappers for your data, and use thrust::reduce to perform the parallel reduction with a single line of C++ code.

Thinking:
- The program will create a large array on the host, then copy it to a thrust::device_vector.
- Thrust provides high-level abstractions; device_vector automatically manages GPU memory.
- The reduce operation can be invoked in one line: thrust::reduce(d_vec.begin(), d_vec.end());
- We'll fill the host vector with a known pattern (e.g., all ones or incremental values) to verify correctness.
- After computing the sum, we'll print it to the console.
- The code will include the necessary Thrust headers and use standard C++ I/O for output.
- No external files or dependencies beyond Thrust and CUDA.
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

int main() {
    const std::size_t N = 1 << 20; // 1,048,576 elements

    // Create a host vector and fill it with values (e.g., 0, 1, 2, ..., N-1)
    thrust::host_vector<int> h_vec(N);
    for (std::size_t i = 0; i < N; ++i) {
        h_vec[i] = static_cast<int>(i);
    }

    // Transfer data to device
    thrust::device_vector<int> d_vec = h_vec; // Implicit copy

    // Perform parallel reduction on the GPU
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

    // Compute expected sum on host for verification
    long long expected = static_cast<long long>(N) * (N - 1) / 2;

    std::cout << "GPU sum = " << sum << std::endl;
    std::cout << "Expected sum = " << expected << std::endl;

    return 0;
}
