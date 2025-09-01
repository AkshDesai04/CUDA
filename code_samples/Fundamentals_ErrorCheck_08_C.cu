/*
Aim of the program: 
Run the version without error checking under `cuda-memcheck`. Observe that `cuda-memcheck` finds the error even if the program itself doesn't report one.

Thinking: 
The objective is to demonstrate a CUDA memory error that is silent in normal execution but will be detected by the CUDA memory checking tool `cuda-memcheck`.  
To achieve this, the program allocates a small array on the GPU, launches a kernel that intentionally writes beyond the bounds of that array, and does **not** perform any CUDA error checking (i.e., it omits calls to `cudaGetLastError`, `cudaMemcpy` error checks, and `cudaDeviceSynchronize` checks).  
This way, when the program is executed normally, it will simply finish and print a completion message. However, when the same binary is run under `cuda-memcheck`, the tool will report the out‑of‑bounds write.  

Key design choices:
- Allocate an array of `int` with 10 elements.
- Launch a kernel with 11 threads in a single block. Each thread writes to `arr[threadIdx.x]`, causing the last thread (index 10) to write past the allocated memory.
- No error checking calls are inserted.
- After the kernel launch, the host prints a simple message and frees the device memory.

The code below implements this simple scenario. When compiled with `nvcc` and run normally, it will exit quietly. When run under `cuda-memcheck` (e.g., `cuda-memcheck ./out_of_bounds_demo`), the tool will detect and report the out‑of‑bounds access.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void write_out_of_bounds(int *arr, int size) {
    int idx = threadIdx.x;
    // Intentionally write beyond the allocated array bounds
    if (idx < size + 1) {  // size is 10, so idx goes up to 10 inclusive
        arr[idx] = idx;   // arr[10] is out of bounds
    }
}

int main() {
    const int N = 10;
    int *d_arr = NULL;
    // Allocate memory for N integers on the device
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    // Launch kernel with N+1 threads to trigger out-of-bounds write
    write_out_of_bounds<<<1, N + 1>>>(d_arr, N);

    // No error checking; just print a message
    printf("Kernel launched. No error checks performed.\n");

    // Free device memory
    cudaFree(d_arr);

    return 0;
}
