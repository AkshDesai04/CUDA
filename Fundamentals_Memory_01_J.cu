```cuda
/*
Aim of the program:
Allocate on host, allocate on device, but forget the cudaMemcpy call. What is in the device memory?

Thinking:
The CUDA runtime does not automatically copy data from host to device. After calling cudaMalloc,
the device memory contains whatever happened to be in that region previously; it is not
guaranteed to be zeroed or contain the host data. In many systems the memory may be
zeroed for security reasons, but this is not a documented behavior and should not be
reliably depended upon. This program demonstrates that by allocating a small array on
the host, allocating an array on the device, *not* copying the host data to the device,
and then copying the device array back to the host to inspect its contents. The output
will show that the device memory holds arbitrary values (often zero, but not guaranteed),
highlighting that forgetting the cudaMemcpy results in uninitialized device memory.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const int N = 10;
    int h_arr[N];
    for (int i = 0; i < N; ++i) h_arr[i] = i + 1;  // Host array initialized to 1..10

    int *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Intentionally omit cudaMemcpy from host to device.

    int h_arr_copy[N];
    err = cudaMemcpy(h_arr_copy, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return -1;
    }

    printf("Contents of device memory after cudaMalloc (no memcpy):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr_copy[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
```