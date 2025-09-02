```c
/*
The callback function must have the signature `void CUDART_CB my_callback(void* user_data)`.

Goal:
The aim of this CUDA program is to demonstrate the use of an asynchronous host callback
(`cudaLaunchHostFunc`) that follows the required signature `void CUDART_CB my_callback(void* user_data)`.
The program launches a simple kernel that adds two arrays. After the kernel finishes,
it schedules the `my_callback` function to run on the host. The callback receives
user data (an integer pointer in this case) and prints a message along with the
value passed to it. This example shows how to pass data to the callback and
ensure it executes after GPU work completes.

Key points considered while writing the code:
1. The CUDA Runtime API provides `cudaLaunchHostFunc` which accepts a function
   with the signature `void CUDART_CB func(void* userData)`. We comply with this.
2. To observe the callback, we allocate an integer, set its value, and pass
   its address to `cudaLaunchHostFunc`. The callback dereferences the pointer.
3. The kernel (`addKernel`) is trivial: it adds two float arrays element-wise.
4. Error checking is performed after every CUDA call using a helper macro
   `CHECK_CUDA(call)`.
5. After launching the host callback, we call `cudaDeviceSynchronize()` to
   ensure all device work and the callback complete before the program exits.
6. The program compiles with `nvcc` and runs on any CUDA-capable device.
7. All includes and definitions are C-compatible and use the CUDA runtime API.

Compile with:
    nvcc -arch=sm_35 callback_demo.cu -o callback_demo

Run:
    ./callback_demo
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                        \
    do {                                                        \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",\
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

/* Simple vector addition kernel */
__global__ void addKernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

/* Callback function matching the required signature */
void CUDART_CB my_callback(void* user_data)
{
    /* user_data is expected to be a pointer to an int */
    int *value = (int *)user_data;
    printf("Callback executed! User data value: %d\n", *value);
}

int main(void)
{
    const int N = 256;
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    /* Prepare user data for the callback */
    int callback_value = 42;
    /* Schedule host callback */
    CHECK_CUDA(cudaLaunchHostFunc(my_callback, &callback_value));

    /* Wait for all device work (including callback) to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify result (optional) */
    int success = 1;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Mismatch at index %d: %f + %f != %f\n",
                   i, h_a[i], h_b[i], h_c[i]);
            success = 0;
            break;
        }
    }
    if (success)
        printf("Result verified: All values correct.\n");

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
```