```c
/*
If you are using CUDA events, wrap `cudaEventCreate`, `cudaEventRecord`, and `cudaEventDestroy`.

The goal of this program is to illustrate how to encapsulate CUDA event functions
within custom wrapper functions that perform error checking and provide a clean
interface for creating, recording, and destroying events.  By wrapping these
functions, we can centralize error handling and simplify the main program
logic.  The wrapper functions (`safeEventCreate`, `safeEventRecord`, and
`safeEventDestroy`) each return a `cudaError_t` status, allowing the caller
to decide how to handle failures.

In addition to the wrappers, the program demonstrates a simple vector addition
kernel.  Two events (`start` and `stop`) are created, recorded before and after
the kernel launch, and then used to compute the elapsed time.  After timing,
the events are destroyed.  All CUDA API calls are checked using the wrapper
functions, ensuring consistent error handling.

The program is self‑contained and can be compiled with `nvcc`:
    nvcc -o cuda_event_wrappers cuda_event_wrappers.cu
It will output the elapsed time for the kernel execution on the GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Wrapper for cudaEventCreate */
cudaError_t safeEventCreate(cudaEvent_t* event) {
    cudaError_t err = cudaEventCreate(event);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(err));
    }
    return err;
}

/* Wrapper for cudaEventRecord */
cudaError_t safeEventRecord(cudaEvent_t event, cudaStream_t stream) {
    cudaError_t err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord failed: %s\n", cudaGetErrorString(err));
    }
    return err;
}

/* Wrapper for cudaEventDestroy */
cudaError_t safeEventDestroy(cudaEvent_t event) {
    cudaError_t err = cudaEventDestroy(event);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaEventDestroy failed: %s\n", cudaGetErrorString(err));
    }
    return err;
}

/* Simple vector addition kernel */
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = (N - i) * 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    safeEventCreate(&start);
    safeEventCreate(&stop);

    /* Record start event */
    safeEventRecord(start, 0);

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Record stop event */
    safeEventRecord(stop, 0);

    /* Wait for kernel to finish */
    cudaDeviceSynchronize();

    /* Compute elapsed time */
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    /* Copy result back to host */
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    /* Cleanup */
    safeEventDestroy(start);
    safeEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```