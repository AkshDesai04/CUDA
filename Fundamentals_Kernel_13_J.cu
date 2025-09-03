/*
Aim of the program: Chain kernel calls. The output of the first kernel (b) becomes the input to a second kernel that, for example, squares every element.

Thinking:
1. The program demonstrates how to launch multiple kernels sequentially where the output of the first kernel is used as the input to the second.
2. We allocate three arrays on the host: an input array, a host result array to hold the final output.
3. On the device we allocate three corresponding arrays: one for the original input, one to store the intermediate result after the first kernel, and one to store the final squared result after the second kernel.
4. The first kernel (firstKernel) takes each element of the input array, multiplies it by 2, and writes it to the intermediate array.
5. The second kernel (secondKernel) takes the intermediate array, squares each element, and writes the result to the final array.
6. After the kernels finish, we copy the final result back to the host and print the first few elements to verify correctness.
7. Error checking is performed after each CUDA API call using a helper macro.
8. The program accepts an optional command line argument to specify the number of elements; otherwise it defaults to 1024.
9. The block size is set to 256 threads per block, and the grid size is computed to cover all elements.
10. Finally, all device memory is freed before the program exits.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call or kernel launch */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",  \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* First kernel: multiplies each element by 2 */
__global__ void firstKernel(const int *d_in, int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_out[idx] = d_in[idx] * 2;
    }
}

/* Second kernel: squares each element */
__global__ void secondKernel(const int *d_in, int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int val = d_in[idx];
        d_out[idx] = val * val;
    }
}

int main(int argc, char *argv[])
{
    /* Determine the number of elements */
    int N = 1024;  // default
    if (argc > 1)
    {
        N = atoi(argv[1]);
        if (N <= 0)
        {
            fprintf(stderr, "Invalid size argument. Must be positive integer.\n");
            return EXIT_FAILURE;
        }
    }

    /* Host arrays */
    int *h_in  = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));
    if (!h_in || !h_out)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array */
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = i;
    }

    /* Device arrays */
    int *d_in  = NULL;
    int *d_mid = NULL;
    int *d_out = NULL;

    CHECK_CUDA(cudaMalloc((void **)&d_in,  N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_mid, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_out, N * sizeof(int)));

    /* Copy input data to device */
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch first kernel */
    firstKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_mid, N);
    CHECK_CUDA(cudaGetLastError());  // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure completion before next kernel

    /* Launch second kernel */
    secondKernel<<<blocksPerGrid, threadsPerBlock>>>(d_mid, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy final result back to host */
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print first 10 results for verification */
    printf("First 10 results after chaining kernels (should be 4*i*i):\n");
    for (int i = 0; i < 10 && i < N; ++i)
    {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_mid));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
