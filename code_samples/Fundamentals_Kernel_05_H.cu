/*
Aim of the program:
Pass two arrays, `in_a` and `in_b`, and one output array `out_c`. Have each thread with global ID `id` compute `out_c[id] = in_a[id] + in_b[id]`.

Thinking:
- The core of the program is a CUDA kernel that receives pointers to the input arrays (`in_a`, `in_b`) and the output array (`out_c`) on the device, as well as the number of elements `N`.  
- Each thread calculates its global index `id` from `blockIdx.x`, `blockDim.x`, and `threadIdx.x`.  
- If `id < N`, the thread performs a single addition: `out_c[id] = in_a[id] + in_b[id];`.  
- The host side is responsible for allocating and initializing the input arrays, copying them to the device, launching the kernel, copying the result back, and cleaning up.  
- Simple error checking is included via a macro that wraps CUDA runtime API calls.  
- A small example is provided in `main()` that demonstrates the workflow with `N = 1024`.  
- The program uses the C API for CUDA (`<cuda_runtime.h>`) and standard C headers (`<stdio.h>`, `<stdlib.h>`).  
- It can be compiled with `nvcc -arch=sm_52 -o vector_add vector_add.cu` (or any architecture that matches the device).  
- The kernel launch uses a block size of 256 threads; the grid size is computed as `(N + threadsPerBlock - 1) / threadsPerBlock`.  
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel that adds two vectors */
__global__ void vector_add_kernel(const float *in_a, const float *in_b, float *out_c, size_t N)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
    {
        out_c[id] = in_a[id] + in_b[id];
    }
}

/* Host function that sets up data, launches kernel, and verifies result */
int main(void)
{
    const size_t N = 1024;          // Number of elements
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_in_a = (float *)malloc(bytes);
    float *h_in_b = (float *)malloc(bytes);
    float *h_out_c = (float *)malloc(bytes);
    if (!h_in_a || !h_in_b || !h_out_c)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input arrays */
    for (size_t i = 0; i < N; ++i)
    {
        h_in_a[i] = (float)i;
        h_in_b[i] = (float)(2 * i);
    }

    /* Allocate device memory */
    float *d_in_a = NULL;
    float *d_in_b = NULL;
    float *d_out_c = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_in_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_in_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out_c, bytes));

    /* Copy input data to device */
    CUDA_CHECK(cudaMemcpy(d_in_a, h_in_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_b, h_in_b, bytes, cudaMemcpyHostToDevice));

    /* Determine execution configuration */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    /* Launch kernel */
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in_a, d_in_b, d_out_c, N);
    CUDA_CHECK(cudaGetLastError());          // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // Wait for kernel to finish

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_out_c, d_out_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify result */
    int errors = 0;
    for (size_t i = 0; i < N; ++i)
    {
        float expected = h_in_a[i] + h_in_b[i];
        if (fabsf(h_out_c[i] - expected) > 1e-5f)
        {
            if (errors < 10) // Print first few errors
            {
                fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n",
                        i, h_out_c[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0)
    {
        printf("Vector addition succeeded. All results are correct.\n");
    }
    else
    {
        printf("Vector addition failed with %d errors.\n", errors);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_in_a));
    CUDA_CHECK(cudaFree(d_in_b));
    CUDA_CHECK(cudaFree(d_out_c));
    free(h_in_a);
    free(h_in_b);
    free(h_out_c);

    return EXIT_SUCCESS;
}
