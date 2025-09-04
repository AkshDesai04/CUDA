/*
 * Aim: To compile code with dynamic parallelism, you need to add `-rdc=true` (relocatable device code) to your `nvcc` flags. Practice this.
 *
 * Thinking:
 * - Dynamic parallelism in CUDA allows a kernel running on the GPU to launch another kernel from the device side.
 * - To enable this feature we must compile with `-rdc=true` (and optionally `-arch=sm_35` or higher, since dynamic parallelism requires compute capability 3.5+).
 * - The simplest demonstration is to have a parent kernel that launches a child kernel which performs a trivial operation (e.g., multiplying each array element by 2).
 * - In this program we allocate an integer array on the GPU, initialize it on the host, copy it to the device, and then launch a single-thread parent kernel.
 * - Inside the parent kernel, the sole thread will launch a child kernel with `N` threads to process the whole array.
 * - After execution, we copy the array back and print the results to verify the child kernel ran.
 * - We include error-checking macros and simple printouts.
 * - Compilation instructions (to be used on the command line):
 *   nvcc -rdc=true -arch=sm_35 -o dyn_parallelism dyn_parallelism.cu
 *   (Adjust -arch to match your GPU if needed.)
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

/* Child kernel: multiplies each element by 2 */
__global__ void childKernel(int *data, int n)
{
    int idx = threadIdx.x;
    if (idx < n)
        data[idx] = data[idx] * 2;
}

/* Parent kernel: launches child kernel from device */
__global__ void parentKernel(int *data, int n)
{
    /* Only one thread performs the launch */
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        /* Launch child kernel with one block and n threads */
        childKernel<<<1, n>>>(data, n);

        /* Synchronize to make sure child completes before parent finishes */
        cudaDeviceSynchronize();
    }
}

int main(void)
{
    const int N = 16;
    int h_data[N];
    int *d_data = NULL;
    size_t size = N * sizeof(int);

    /* Initialize host data */
    for (int i = 0; i < N; ++i)
        h_data[i] = i + 1;

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void **)&d_data, size));

    /* Copy data to device */
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    /* Launch parent kernel with one block and one thread */
    parentKernel<<<1, 1>>>(d_data, N);

    /* Check for any errors launching kernel */
    CHECK_CUDA(cudaGetLastError());

    /* Wait for GPU to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Result after dynamic parallelism (each element multiplied by 2):\n");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_data[i]);
    printf("\n");

    /* Clean up */
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
