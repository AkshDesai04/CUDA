```c
/*
 * Aim of the program:
 * Use `sizeof()` on the host and device to print the size of the struct and verify they are the same.
 *
 * Thinking:
 * 1. Define a simple POD struct `MyStruct` with a few members of different types.
 * 2. On the host, compute `sizeof(MyStruct)` and print it.
 * 3. On the device, compute the same using `sizeof(MyStruct)` inside a kernel.
 * 4. To verify equality, we will:
 *    a) Use device `printf` to print the size inside the kernel.
 *    b) Also write the size into a global device variable.
 * 5. Copy the device value back to the host using `cudaMemcpy` and compare it with the host size.
 * 6. Report whether the sizes match.
 *
 * This demonstrates that the compiler applies the same rules for `sizeof` on host and device,
 * and that the struct layout is consistent across host and device code.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Define a simple struct with different data types */
struct MyStruct {
    int    a;
    float  b;
    double c;
};

/* Device global variable to hold the size of MyStruct */
__device__ size_t d_struct_size = 0;

/* Kernel to compute and print the size of MyStruct on the device */
__global__ void compute_size_on_device()
{
    /* Compute size on device */
    size_t size = sizeof(MyStruct);

    /* Write to device global variable for host retrieval */
    d_struct_size = size;

    /* Print size using device printf (requires device compute capability >= 2.0) */
    printf("Size of MyStruct on device: %zu bytes\n", size);
}

int main()
{
    /* Host compute size */
    size_t h_struct_size = sizeof(MyStruct);
    printf("Size of MyStruct on host: %zu bytes\n", h_struct_size);

    /* Launch kernel */
    compute_size_on_device<<<1, 1>>>();
    cudaDeviceSynchronize();

    /* Copy device size back to host */
    size_t d_struct_size_host = 0;
    cudaError_t err = cudaMemcpyFromSymbol(&d_struct_size_host, d_struct_size, sizeof(size_t), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Verify sizes are equal */
    if (h_struct_size == d_struct_size_host) {
        printf("Verification succeeded: Host and device sizes match.\n");
    } else {
        printf("Verification failed: Host size = %zu, Device size = %zu\n", h_struct_size, d_struct_size_host);
    }

    return 0;
}
```