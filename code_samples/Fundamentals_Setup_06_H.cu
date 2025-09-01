/*
Aim:
Query and print the surface alignment requirement in bytes.

Thinking:
To retrieve the surface alignment requirement on a CUDA device, the CUDA Runtime API provides the function cudaGetSurfaceAlignment. This function writes the alignment in bytes to a size_t pointer supplied by the caller and returns a cudaError_t indicating success or failure. The alignment requirement is architecture dependent; some GPUs may require surfaces to be aligned to 256 bytes, for instance.  

I will write a simple CUDA host program that:
1. Includes the necessary headers (<cstdio> for printf and <cuda_runtime.h> for CUDA API).
2. Calls cudaGetSurfaceAlignment, checking the returned error code with cudaGetErrorString for diagnostics.
3. If the call succeeds, prints the alignment value in bytes.
4. If it fails, reports the error and exits with a non‑zero status.

The program is intentionally minimal, purely demonstrating the query and print functionality. It can be compiled with nvcc (e.g., `nvcc -o surface_alignment surface_alignment.cu`) and run on any system with CUDA support.*/

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    size_t alignment = 0;
    cudaError_t err = cudaGetSurfaceAlignment(&alignment);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetSurfaceAlignment failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Surface alignment requirement: %zu bytes\n", alignment);
    return 0;
}
