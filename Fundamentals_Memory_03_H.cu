```cuda
/*
What happens if the source and destination pointers for a `cudaMemcpyDeviceToDevice` overlap? (Behavior is undefined, use `cudaMemmove` if needed).

Goal of this program:
1. Allocate a small array on the device.
2. Initialize it with a known sequence of values.
3. Perform a device-to-device copy where the source and destination ranges overlap.
4. Copy the data back to the host and print the results to observe what actually happened.
5. For comparison, perform the same overlapping copy using `cudaMemmove`, which is defined for overlapping regions.
6. Print the results of the `cudaMemmove` operation as well.
7. Discuss that the result from the overlapping `cudaMemcpyDeviceToDevice` is not guaranteed to be correct, while `cudaMemmove` behaves as expected.

Why this matters:
The CUDA runtime documentation explicitly states that `cudaMemcpyDeviceToDevice` behaves like `memcpy` on the host, which does not support overlapping regions. The behavior is therefore undefined and could produce corrupted data or no change at all. Using `cudaMemmove` is the correct way to safely move data on the device when overlaps are possible.

The code below implements these steps with comprehensive error checking and prints both the original and the overlapped results to highlight the difference between the two APIs.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error in file '" << __FILE__                   \
                      << "' in line " << __LINE__ << ": "                    \
                      << cudaGetErrorString(err) << std::endl;               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main() {
    const int N = 10;
    const int size = N * sizeof(int);

    int h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = i; // 0 1 2 3 4 5 6 7 8 9

    int *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    std::cout << "Original device data: ";
    for (int i = 0; i < N; ++i) std::cout << h_data[i] << " ";
    std::cout << std::endl;

    /* ------------------------------------------------------------------ */
    /* Overlapping copy using cudaMemcpyDeviceToDevice (undefined behavior) */
    /* ------------------------------------------------------------------ */
    // Let's copy 5 elements from index 0 to index 2 (dest overlaps source)
    int *src = d_data + 0;        // start at element 0
    int *dst = d_data + 2;        // start at element 2
    int count = 5 * sizeof(int); // 5 integers

    // Perform the overlapping copy
    CHECK_CUDA(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));

    // Copy back to host to inspect result
    int h_result1[N];
    CHECK_CUDA(cudaMemcpy(h_result1, d_data, size, cudaMemcpyDeviceToHost));

    std::cout << "\nAfter cudaMemcpyDeviceToDevice with overlap:" << std::endl;
    for (int i = 0; i < N; ++i) std::cout << h_result1[i] << " ";
    std::cout << std::endl;

    /* ------------------------------------------------------------------ */
    /* Reset device memory to original values */
    /* ------------------------------------------------------------------ */
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    /* ------------------------------------------------------------------ */
    /* Overlapping copy using cudaMemmove (defined behavior) */
    /* ------------------------------------------------------------------ */
    // cudaMemmove is available as cudaMemmove in the CUDA Runtime
    CHECK_CUDA(cudaMemmove(dst, src, count));

    int h_result2[N];
    CHECK_CUDA(cudaMemcpy(h_result2, d_data, size, cudaMemcpyDeviceToHost));

    std::cout << "\nAfter cudaMemmove with overlap:" << std::endl;
    for (int i = 0; i < N; ++i) std::cout << h_result2[i] << " ";
    std::cout << std::endl;

    /* ------------------------------------------------------------------ */
    /* Clean up */
    /* ------------------------------------------------------------------ */
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```