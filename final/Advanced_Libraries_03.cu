/*
Use the Thrust library to find the sum of a large array of integers on the GPU. Include the Thrust headers, create `thrust::device_vector` wrappers for your data, and use `thrust::reduce` to perform the parallel reduction with a single line of C++ code.

Thinking:
- To satisfy the requirement, I'll create a CUDA C++ source file with a .cu extension.
- Include necessary Thrust headers: thrust/host_vector.h, thrust/device_vector.h, thrust/reduce.h.
- Include <iostream> for output.
- In main, define a large size, e.g., 10 million elements. Initialize a host_vector with deterministic data (e.g., each element equal to its index mod 100) so we can verify the sum.
- Copy to device_vector via assignment.
- Use thrust::reduce with a single line: int sum = thrust::reduce(d_vec.begin(), d_vec.end()); This uses default initial value 0 and plus operator.
- Print sum. Optionally compute expected sum on host for verification.
- The program uses thrust to perform GPU reduction, ensuring that the code compiles with nvcc.
- No other text or output besides the .cu file content is provided.
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

int main() {
    const size_t N = 10000000; // 10 million elements

    // Create and initialize host vector
    thrust::host_vector<int> h_vec(N);
    for (size_t i = 0; i < N; ++i) {
        h_vec[i] = static_cast<int>(i % 100); // deterministic pattern
    }

    // Transfer data to device
    thrust::device_vector<int> d_vec = h_vec;

    // Perform parallel reduction using a single line
    int sum = thrust::reduce(d_vec.begin(), d_vec.end()); // default initial value 0, plus

    std::cout << "Sum of array elements: " << sum << std::endl;

    return 0;
}
