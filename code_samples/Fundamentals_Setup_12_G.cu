/*
Pass a host pointer (e.g., from `malloc`) directly to a kernel that expects a device pointer.
The aim of this program is to demonstrate what happens when you mistakenly pass a host pointer
to a CUDA kernel that expects a device pointer.  In CUDA, device kernels execute on the GPU
and can only dereference pointers that refer to memory allocated on the device (e.g., via
cudaMalloc).  If a host pointer (obtained from malloc, new, or stack) is passed directly to a
kernel, the GPU will attempt to read/write from an invalid memory location, resulting in
runtime errors or undefined behavior.

To illustrate this, the program allocates an array on the host, copies it to device memory,
and then intentionally launches a kernel with the host pointer instead of the device
pointer.  After the kernel launch we query cudaGetLastError() to capture the failure.
The kernel itself is trivial – it simply writes twice the input value into an output
array.  We also print the result only if the kernel succeeded; otherwise we report the
error.  This program is intentionally incorrect on purpose to show the error that
occurs when mixing host and device pointers incorrectly.

The code demonstrates:
1. Allocation of host memory with malloc.
2. Allocation of device memory with cudaMalloc.
3. Copying data from host to device with cudaMemcpy.
4. Incorrect kernel invocation using a host pointer.
5. Error checking with cudaGetLastError().
6. Cleaning up all allocated resources.

Compile with:
    nvcc -o host_pointer_demo host_pointer_demo.cu
Run with:
    ./host_pointer_demo
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumKernel(const int *a, int *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] * 2;  // simple operation
    }
}

int main(void)
{
    const int N = 10;
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_c = (int *)malloc(N * sizeof(int));

    /* Initialize host input */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
    }

    /* Allocate device memory */
    int *d_a, *d_c;
    cudaError_t err;

    err = cudaMalloc((void **)&d_a, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_a: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void **)&d_c, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_c: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        return -1;
    }

    /* Copy input from host to device */
    err = cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_c);
        return -1;
    }

    /* Intentionally pass host pointer h_a to kernel expecting device pointer */
    sumKernel<<<1, N>>>(h_a, d_c, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed (expected due to host pointer): %s\n", cudaGetErrorString(err));
    } else {
        /* Wait for kernel to finish */
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        } else {
            /* Copy result back to host */
            err = cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy DeviceToHost failed: %s\n", cudaGetErrorString(err));
            } else {
                printf("Results: ");
                for (int i = 0; i < N; ++i) {
                    printf("%d ", h_c[i]);
                }
                printf("\n");
            }
        }
    }

    /* Clean up */
    cudaFree(d_a);
    cudaFree(d_c);
    free(h_a);
    free(h_c);

    return 0;
}
