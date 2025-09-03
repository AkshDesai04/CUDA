```cpp
/*
Aim: Explain why the pitch returned by cudaMallocPitch may be larger than width * sizeof(element). (Hint: alignment for coalescing)

Thinking:
The pitch returned by cudaMallocPitch is the number of bytes between the start of two consecutive rows in a 2D array. 
When we ask for a 2D allocation of width * sizeof(element) bytes per row, the driver might adjust the pitch upward
to satisfy hardware alignment constraints.  In particular, many GPUs require that accesses to consecutive elements 
in a warp are aligned to a 128‑byte boundary to achieve maximum memory coalescing.  If width * sizeof(element) is not a 
multiple of 128, the GPU will pad each row so that the start of the next row aligns on the next 128‑byte boundary.  
Consequently, the pitch can be larger than the requested width * sizeof(element).  This padding does not affect the 
logical 2‑D shape of the array but improves memory access patterns and performance.  The same idea applies to
other alignment constraints (e.g., 32‑byte alignment for single‑thread loads or 512‑byte alignment for texture
accesses).  cudaMallocPitch handles these details internally, returning a pitch that satisfies the GPU’s
coalescing and alignment requirements while still allowing the user to treat the allocation as a rectangular
array with the requested logical width and height.

Below is a simple CUDA C program that allocates a pitched 2D array, prints the requested width, the actual pitch, 
and shows that the pitch may be larger than width * sizeof(element).  The program also demonstrates how the 
difference arises due to alignment.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    const size_t width = 3;            // number of elements per row
    const size_t height = 4;           // number of rows
    const size_t elemSize = sizeof(float); // size of one element

    // Requested row size in bytes
    const size_t requestedRowSize = width * elemSize;

    float *d_ptr = nullptr;
    size_t pitch = 0;

    // Allocate pitched memory
    cudaError_t err = cudaMallocPitch(&d_ptr, &pitch, requestedRowSize, height);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Requested row size (bytes): %zu\n", requestedRowSize);
    printf("Allocated pitch (bytes):   %zu\n", pitch);

    if (pitch > requestedRowSize) {
        printf("Pitch is larger than requested row size.\n");
        printf("The difference is %zu bytes, likely due to alignment requirements for coalesced memory accesses.\n",
               pitch - requestedRowSize);
    } else {
        printf("Pitch equals requested row size.\n");
    }

    // Clean up
    cudaFree(d_ptr);
    return 0;
}
```