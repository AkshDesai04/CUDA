/*
Use `nvidia-smi` to watch the GPU memory usage while your program runs.
Step through with a debugger to see the memory increase after `cudaMalloc` and decrease after `cudaFree`.

Thinking: This program allocates a 100 MB block on the GPU, prints GPU memory usage before and after allocation, then frees it and prints again. By running nvidia-smi and a debugger, you can observe the memory usage increase and decrease accordingly. Using cudaMemGetInfo allows us to programmatically see the free/total memory values. The code includes error checking for robustness. The aim is to demonstrate CUDA memory management and to provide an example that can be debugged step-by-step.
*/

#include <stdio.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t size = 100 * 1024 * 1024;  // 100 MB
    int *d_ptr = NULL;

    size_t freeBefore, totalBefore;
    checkCudaError(cudaMemGetInfo(&freeBefore, &totalBefore), "cudaMemGetInfo before");

    printf("GPU memory before allocation: %zu / %zu bytes free / total.\n", freeBefore, totalBefore);

    cudaError_t err = cudaMalloc((void**)&d_ptr, size);
    checkCudaError(err, "cudaMalloc");

    size_t freeAfterAlloc, totalAfterAlloc;
    checkCudaError(cudaMemGetInfo(&freeAfterAlloc, &totalAfterAlloc), "cudaMemGetInfo after allocation");

    printf("Allocated %zu bytes on GPU.\n", size);
    printf("GPU memory after allocation: %zu / %zu bytes free / total.\n", freeAfterAlloc, totalAfterAlloc);

    cudaError_t errFree = cudaFree(d_ptr);
    checkCudaError(errFree, "cudaFree");

    size_t freeAfterFree, totalAfterFree;
    checkCudaError(cudaMemGetInfo(&freeAfterFree, &totalAfterFree), "cudaMemGetInfo after free");

    printf("Freed GPU memory.\n");
    printf("GPU memory after free: %zu / %zu bytes free / total.\n", freeAfterFree, totalAfterFree);

    return 0;
}
