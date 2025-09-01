```cpp
/*
 Aim of the program: Try to use `cudaMemcpyFromSymbol` to read the value of the constant variable back to the host.
 
 Thinking process:
 1. We want to demonstrate that a constant memory variable declared on the device can be written from the host and later read back.
 2. `__constant__` variables are read‑only from the device, but they can be modified from the host via `cudaMemcpyToSymbol`.
 3. After setting the constant value, we will copy it back to the host using `cudaMemcpyFromSymbol`.
 4. We will use a simple `int` constant to keep the example straightforward.
 5. The program will:
    - Declare a constant memory variable `__constant__ int d_constVal`.
    - Define a host integer `h_valToSet = 42;`.
    - Copy `h_valToSet` to `d_constVal` using `cudaMemcpyToSymbol`.
    - Define another host integer `h_valReadBack`.
    - Copy the value from `d_constVal` back to `h_valReadBack` using `cudaMemcpyFromSymbol`.
    - Print both the original value set and the value read back to verify correctness.
 6. We'll add basic CUDA error checking for robustness.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Constant memory variable on the device
__constant__ int d_constVal;

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main()
{
    const int h_valToSet = 42;
    int h_valReadBack = 0;

    // Write value to constant memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_constVal, &h_valToSet, sizeof(int)));

    // Read value back from constant memory
    CHECK_CUDA(cudaMemcpyFromSymbol(&h_valReadBack, d_constVal, sizeof(int)));

    // Verify the round‑trip
    printf("Value set on device constant memory: %d\n", h_valToSet);
    printf("Value read back from device constant memory: %d\n", h_valReadBack);

    return 0;
}
```