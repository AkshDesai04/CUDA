```cuda
/* 
Write a CHECK_KERNEL_LAUNCH() macro that calls cudaPeekAtLastError() and cudaDeviceSynchronize() and reports any errors.

Thinking:
- The macro should be usable immediately after launching a CUDA kernel to catch any launch errors.
- It will first call cudaPeekAtLastError() to obtain the error status from the last kernel launch.
- If an error is detected, we report it using cudaGetErrorString() and the source file/line where the macro was invoked.
- Then the macro will call cudaDeviceSynchronize() to ensure all pending GPU work is finished and capture any errors that occur during execution.
- If cudaDeviceSynchronize() returns an error, we report that as well.
- Using a do{...}while(0) construct makes the macro safe to use in all contexts (e.g., within if statements without braces).
- The macro will print errors to stderr for visibility.
- The code below also includes a simple example kernel and a main function to demonstrate the macro in action.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check kernel launch and device synchronization errors */
#define CHECK_KERNEL_LAUNCH()                                \
    do {                                                    \
        cudaError_t err = cudaPeekAtLastError();            \
        if (err != cudaSuccess) {                          \
            fprintf(stderr, "CUDA kernel launch error: %s\n", \
                    cudaGetErrorString(err));              \
            fprintf(stderr, "    at %s:%d\n", __FILE__, __LINE__); \
        }                                                   \
        err = cudaDeviceSynchronize();                      \
        if (err != cudaSuccess) {                          \
            fprintf(stderr, "CUDA device synchronization error: %s\n", \
                    cudaGetErrorString(err));              \
            fprintf(stderr, "    at %s:%d\n", __FILE__, __LINE__); \
        }                                                   \
    } while (0)

/* Example kernel that does nothing */
__global__ void dummyKernel(int *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = a[idx] + 1; // simple operation to avoid compiler optimizations
}

int main(void) {
    const int N = 1024;
    int *d_a, *h_a;
    h_a = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_a[i] = i;

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    /* Launch kernel */
    dummyKernel<<<(N + 255)/256, 256>>>(d_a);
    CHECK_KERNEL_LAUNCH();   // <--- This macro checks for errors

    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    /* Simple check to ensure kernel executed */
    if (h_a[0] == 1) {
        printf("Kernel executed successfully.\n");
    } else {
        printf("Unexpected result.\n");
    }

    cudaFree(d_a);
    free(h_a);
    return 0;
}
```