```cpp
/* 
Aim: Define a struct with many members. Time the difference between copying the whole struct vs. using `cudaMemcpy` on individual members (the former is much better).

Thinking:
- Create a struct `MyStruct` with diverse member types and several fields to mimic a realistic data structure.
- Use host and device memory allocation for a single instance of this struct.
- Perform two copy strategies:
  1. Copy the entire struct in a single `cudaMemcpy` call.
  2. Copy each member separately using `cudaMemcpy`, computing device addresses via member offsets.
- Time each strategy using CUDA events for accurate measurement on the device.
- Verify correctness by copying the data back and comparing.
- Report the elapsed times and illustrate that a single full-struct copy is more efficient.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <cstddef>   // for offsetof
#include <cstring>   // for memset

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error: %s (err %d) at line %d\n",             \
                    cudaGetErrorString(err), err, __LINE__);                    \
            return -1;                                                          \
        }                                                                       \
    } while (0)

// Define a struct with many members of different types
struct MyStruct {
    int    a;
    float  b;
    double c;
    char   d[64];
    int    e;
    float  f;
    double g;
    char   h[128];
    int    i;
    float  j;
};

// Helper function to initialize the struct with known values
void initStruct(MyStruct& s) {
    s.a = 42;
    s.b = 3.14f;
    s.c = 2.718281828;
    std::memset(s.d, 'A', sizeof(s.d));
    s.e = -7;
    s.f = 1.618f;
    s.g = 0.5772156649;
    std::memset(s.h, 'B', sizeof(s.h));
    s.i = 123456789;
    s.j = -9.81f;
}

// Helper function to compare two structs
bool compareStructs(const MyStruct& a, const MyStruct& b) {
    return std::memcmp(&a, &b, sizeof(MyStruct)) == 0;
}

int main() {
    // Host struct initialization
    MyStruct h_struct;
    initStruct(h_struct);

    // Allocate device memory for the struct
    MyStruct *d_struct = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_struct, sizeof(MyStruct)));

    // Create CUDA events for timing
    cudaEvent_t startWhole, stopWhole;
    cudaEvent_t startParts, stopParts;
    CHECK_CUDA(cudaEventCreate(&startWhole));
    CHECK_CUDA(cudaEventCreate(&stopWhole));
    CHECK_CUDA(cudaEventCreate(&startParts));
    CHECK_CUDA(cudaEventCreate(&stopParts));

    // ----- Strategy 1: Copy the whole struct in one call -----
    CHECK_CUDA(cudaEventRecord(startWhole));
    CHECK_CUDA(cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stopWhole));
    CHECK_CUDA(cudaEventSynchronize(stopWhole));

    // ----- Strategy 2: Copy each member individually -----
    // Compute base pointer for device struct
    char* d_base = reinterpret_cast<char*>(d_struct);
    // Record start
    CHECK_CUDA(cudaEventRecord(startParts));

    // Copy each member using offset
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, a), &h_struct.a, sizeof(h_struct.a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, b), &h_struct.b, sizeof(h_struct.b), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, c), &h_struct.c, sizeof(h_struct.c), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, d), &h_struct.d, sizeof(h_struct.d), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, e), &h_struct.e, sizeof(h_struct.e), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, f), &h_struct.f, sizeof(h_struct.f), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, g), &h_struct.g, sizeof(h_struct.g), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, h), &h_struct.h, sizeof(h_struct.h), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, i), &h_struct.i, sizeof(h_struct.i), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_base + offsetof(MyStruct, j), &h_struct.j, sizeof(h_struct.j), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(stopParts));
    CHECK_CUDA(cudaEventSynchronize(stopParts));

    // Compute elapsed times
    float msWhole = 0.0f, msParts = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&msWhole, startWhole, stopWhole));
    CHECK_CUDA(cudaEventElapsedTime(&msParts, startParts, stopParts));

    // Verify correctness by copying back to host
    MyStruct h_struct_back;
    CHECK_CUDA(cudaMemcpy(&h_struct_back, d_struct, sizeof(MyStruct), cudaMemcpyDeviceToHost));
    if (!compareStructs(h_struct, h_struct_back)) {
        fprintf(stderr, "Data mismatch after copy back from device!\n");
    } else {
        printf("Verification passed: data on device matches host.\n");
    }

    // Print timing results
    printf("Whole struct copy time:      %.5f ms\n", msWhole);
    printf("Individual members copy time: %.5f ms\n", msParts);
    if (msWhole > 0) {
        printf("Speedup (whole struct / parts): %.2fx\n", msParts / msWhole);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_struct));
    CHECK_CUDA(cudaEventDestroy(startWhole));
    CHECK_CUDA(cudaEventDestroy(stopWhole));
    CHECK_CUDA(cudaEventDestroy(startParts));
    CHECK_CUDA(cudaEventDestroy(stopParts));

    return 0;
}
```