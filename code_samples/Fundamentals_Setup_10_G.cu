```cpp
/* 
Aim of the program: Query and print the `regsPerBlock` property. 

Thinking process:
1. Use CUDA runtime API to query device properties.
2. Call cudaGetDevice to get the current device ID.
3. Declare a cudaDeviceProp structure and fill it using cudaGetDeviceProperties.
4. The property `regsPerBlock` is an integer field within cudaDeviceProp that specifies the number of registers per block.
5. Print this value to the console.
6. Include simple error checking after each CUDA call.
7. Keep the program minimal and self‑contained; it only prints the requested property and exits.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                  \
        }                                                          \
    } while (0)

int main(void) {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("regsPerBlock: %d\n", prop.regsPerBlock);

    return EXIT_SUCCESS;
}
```