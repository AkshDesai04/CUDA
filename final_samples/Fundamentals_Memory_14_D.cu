/*
The error reported by `cuda-memcheck` is an asynchronous error. Show that without `cuda-memcheck`, the program may run and exit without any visible error, potentially producing wrong results.

The following program intentionally performs an out‑of‑bounds write on the device. 
It allocates a contiguous block of 2*N integers and treats the first N as one array (`arr1`) 
and the second N as another (`arr2`). The kernel is launched with 2*N threads, but only the
first N indices are valid for `arr1`. Threads with indices >= N write past the end of
`arr1`, which is the beginning of `arr2`. This corrupts the data in `arr2`. 

Because the kernel launch is asynchronous, the program does not immediately detect
the error. We intentionally do not call `cudaDeviceSynchronize()` or check the return
value of the kernel launch. The subsequent `cudaMemcpy` operations will complete the
copy, but by that time the data may already be corrupted. When the program runs
without `cuda-memcheck`, it will print seemingly normal output for `arr1` and
possibly nonsensical or partially correct values for `arr2`, yet no error will be
reported. Running the same code under `cuda-memcheck` will flag the out‑of‑bounds
write as an asynchronous error.

Compile with:
    nvcc -o async_error async_error.cu

Run normally:
    ./async_error

Run under cuda-memcheck:
    cuda-memcheck ./async_error
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10   // Size of each logical array

__global__ void faultyKernel(int *arr, int totalThreads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < totalThreads)
    {
        // First N elements are logical array1; next N are array2
        arr[idx] = idx;              // Intended write
        arr[idx] = idx + 1000;       // Overwrite to demonstrate corruption
    }
}

int main()
{
    int *d_data;
    int *h_arr2 = (int*)malloc(N * sizeof(int));

    // Allocate contiguous device memory for both arrays
    cudaMalloc((void**)&d_data, 2 * N * sizeof(int));

    // Launch kernel with 2*N threads, all writing to d_data
    int totalThreads = 2 * N;
    faultyKernel<<<1, totalThreads>>>(d_data, totalThreads);
    // No synchronization or error checking after launch

    // Copy back the second logical array (arr2) which may be corrupted
    cudaMemcpy(h_arr2, d_data + N, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Array2 after kernel launch (may be corrupted):\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_arr2[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_data);
    free(h_arr2);

    return 0;
}
