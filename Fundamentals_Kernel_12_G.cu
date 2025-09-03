/*
Read the desired number of threads from the command line and dynamically calculate the grid/block `dim3` variables at runtime.

Thinking:
1. The program will accept one command-line argument: the desired number of threads to launch.
2. We need to determine a suitable block size. CUDA allows up to 1024 threads per block on many devices, so we'll use the minimum of the requested number and 1024.
3. The number of blocks will be calculated as `(threads + blockSize - 1) / blockSize` to cover all threads.
4. A simple kernel will be launched that writes the global thread index into an array. This demonstrates that each thread has executed.
5. After the kernel finishes, the host will copy back the array and print the first few elements to confirm correct execution.
6. All CUDA API calls will be checked for errors. 
7. The program is written in C (CUDA C), outputting a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void fillArray(int *arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = idx;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int desired_threads = atoi(argv[1]);
    if (desired_threads <= 0) {
        fprintf(stderr, "Number of threads must be positive.\n");
        return EXIT_FAILURE;
    }

    /* Determine block and grid sizes */
    int threads_per_block = (desired_threads < 1024) ? desired_threads : 1024;
    int num_blocks = (desired_threads + threads_per_block - 1) / threads_per_block;

    dim3 blockDim(threads_per_block, 1, 1);
    dim3 gridDim(num_blocks, 1, 1);

    printf("Launching %d threads using %d blocks of %d threads each.\n",
           desired_threads, num_blocks, threads_per_block);

    /* Allocate memory on device */
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void **)&d_arr, desired_threads * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    /* Launch kernel */
    fillArray<<<gridDim, blockDim>>>(d_arr, desired_threads);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return EXIT_FAILURE;
    }

    /* Copy results back to host */
    int *h_arr = (int *)malloc(desired_threads * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Host allocation failed.\n");
        cudaFree(d_arr);
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(h_arr, d_arr, desired_threads * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        cudaFree(d_arr);
        return EXIT_FAILURE;
    }

    /* Print first few elements to verify */
    printf("First 10 elements of array:\n");
    int print_count = (desired_threads < 10) ? desired_threads : 10;
    for (int i = 0; i < print_count; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Clean up */
    free(h_arr);
    cudaFree(d_arr);

    return EXIT_SUCCESS;
}
