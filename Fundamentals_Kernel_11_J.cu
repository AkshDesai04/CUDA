```cpp
/* 
Aim of the program: Define a struct with many members. Time the difference between copying the whole struct vs. using cudaMemcpy on individual members (the former is much better).

Thinking:
1. Create a struct (`MyStruct`) that contains a variety of data types and a few arrays to make the size non‑trivial.
2. Allocate a host instance and a device instance of the struct.
3. Fill the host struct with sample data.
4. Use CUDA events to time:
   - a single `cudaMemcpy` that copies the entire struct at once (host -> device).
   - a loop that copies each member separately.  To do this we use `offsetof` to find each member’s offset and `sizeof` for its size, then perform a `cudaMemcpy` for that portion.
5. Repeat each method many times (e.g., 10 000 iterations) so that the timings are stable and dominated by data transfer, not by the overhead of launching the copy operations.
6. Compute and print the average time per iteration for both strategies.
7. Optionally copy the struct back to the host and verify that the data was transferred correctly (not required for timing but useful for correctness).
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstddef>  // for offsetof

// Helper macro for CUDA error checking
#define CUDA_CHECK(err)                                        \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl;     \
        exit(EXIT_FAILURE);                                    \
    }

// Define a struct with many members
struct MyStruct {
    int       a;
    float     b;
    double    c;
    char      d[32];
    bool      e;
    long long f;
    short     g;
    unsigned  h;
    float     arr[16];
    char      name[64];
};

// Number of iterations for timing
const int NUM_ITER = 10000;

int main() {
    // Allocate host and device structs
    MyStruct h_struct;
    MyStruct* d_struct;
    CUDA_CHECK(cudaMalloc((void**)&d_struct, sizeof(MyStruct)));

    // Initialize host struct with dummy data
    h_struct.a = 42;
    h_struct.b = 3.14f;
    h_struct.c = 2.71828;
    for (int i = 0; i < 32; ++i) h_struct.d[i] = 'A' + i;
    h_struct.e = true;
    h_struct.f = 123456789012345LL;
    h_struct.g = 7;
    h_struct.h = 0xDEADBEEF;
    for (int i = 0; i < 16; ++i) h_struct.arr[i] = float(i) * 1.1f;
    for (int i = 0; i < 64; ++i) h_struct.name[i] = 'N' + i;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------- Full struct copy ---------------------
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_ITER; ++i) {
        CUDA_CHECK(cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_full;
    CUDA_CHECK(cudaEventElapsedTime(&ms_full, start, stop));
    ms_full /= NUM_ITER;  // average per iteration

    // ----------------- Individual member copy -----------------
    // Prepare arrays of offsets and sizes for each member
    const size_t offsets[] = {
        offsetof(MyStruct, a),
        offsetof(MyStruct, b),
        offsetof(MyStruct, c),
        offsetof(MyStruct, d),
        offsetof(MyStruct, e),
        offsetof(MyStruct, f),
        offsetof(MyStruct, g),
        offsetof(MyStruct, h),
        offsetof(MyStruct, arr),
        offsetof(MyStruct, name)
    };
    const size_t sizes[] = {
        sizeof(h_struct.a),
        sizeof(h_struct.b),
        sizeof(h_struct.c),
        sizeof(h_struct.d),
        sizeof(h_struct.e),
        sizeof(h_struct.f),
        sizeof(h_struct.g),
        sizeof(h_struct.h),
        sizeof(h_struct.arr),
        sizeof(h_struct.name)
    };
    const int num_members = sizeof(offsets) / sizeof(offsets[0]);

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_ITER; ++i) {
        for (int m = 0; m < num_members; ++m) {
            CUDA_CHECK(cudaMemcpy(
                (char*)d_struct + offsets[m],
                (char*)&h_struct + offsets[m],
                sizes[m],
                cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_individual;
    CUDA_CHECK(cudaEventElapsedTime(&ms_individual, start, stop));
    ms_individual /= NUM_ITER;  // average per iteration

    // Output results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average time per iteration:\n";
    std::cout << "  Full struct copy:     " << ms_full     << " ms\n";
    std::cout << "  Individual members:   " << ms_individual << " ms\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_struct));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```