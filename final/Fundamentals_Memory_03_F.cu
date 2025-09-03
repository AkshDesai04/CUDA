```cpp
/*
Allocate `d_a` and `d_b`. Use `cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice)` to initialize `d_a`. What is the error? Change `h_a` to `d_a`. What is the error? Explain why the kind parameter is critical.

My thinking:
- The goal is to illustrate two common mistakes when using cudaMemcpy.
- First, the user incorrectly copies host data into the wrong device pointer (`d_b`) while intending to initialize `d_a`. The copy succeeds from the host viewpoint, but `d_a` remains uninitialized, leading to wrong results downstream. The error is logical, not a runtime error, because the source and destination pointers are both valid memory addresses (host to device), so cudaMemcpy will not report an error, but the data will be wrong.
- Second, the user then tries to copy from device to device by passing a device pointer (`d_a`) as the source but still using the kind `cudaMemcpyHostToDevice`. The runtime expects a host pointer for the source, so it will flag an error (invalid value). The kind parameter tells the runtime which memory space each argument refers to. Using the wrong kind leads to either undefined behaviour or explicit errors.
- This demonstrates that the `kind` parameter is critical: it specifies the direction of the copy and, implicitly, which side (host or device) each pointer refers to. Mismatching this can result in data corruption or runtime errors.

The code below allocates two device arrays (`d_a`, `d_b`) and a host array (`h_a`). It then performs the two erroneous copies and prints any CUDA error codes. Finally, it shows the correct usage for each operation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    const int N = 10;
    const size_t size = N * sizeof(float);

    /* Host data */
    float h_a[N];
    for (int i = 0; i < N; ++i) h_a[i] = (float)i;

    /* Device pointers */
    float *d_a = nullptr;
    float *d_b = nullptr;

    cudaError_t err;

    /* Allocate device memory */
    err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_a failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        return 1;
    }

    /* --------------------------------------------------------------------
     * Mistake 1: Copy host data into d_b but claim to initialize d_a.
     * The copy operation is legal (Host -> Device) but writes to the wrong
     * device buffer. No CUDA error is reported; the data in d_a remains
     * uninitialized (contains garbage). This is a logical error.
     * -------------------------------------------------------------------- */
    err = cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice): %s\n",
                cudaGetErrorString(err));
    } else {
        printf("cudaMemcpy(d_b, h_a, size, cudaMemcpyHostToDevice) succeeded.\n");
        printf("d_a remains uninitialized; data will be wrong if used.\n");
    }

    /* --------------------------------------------------------------------
     * Mistake 2: Copy from d_a to d_b but still use cudaMemcpyHostToDevice.
     * The runtime expects a host pointer as the source, but receives a device
     * pointer. This leads to a cudaErrorInvalidValue.
     * -------------------------------------------------------------------- */
    err = cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice): %s\n",
                cudaGetErrorString(err));
    } else {
        printf("Unexpected success: cudaMemcpy(d_b, d_a, size, cudaMemcpyHostToDevice) succeeded.\n");
    }

    /* --------------------------------------------------------------------
     * Correct usage:
     *   1. Initialize d_a from host: cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice)
     *   2. Copy from d_a to d_b (device-to-device): cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice)
     * -------------------------------------------------------------------- */
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Correct copy d_a <- h_a failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Correctly copied h_a to d_a.\n");
    }

    err = cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Correct copy d_b <- d_a failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Correctly copied d_a to d_b (device-to-device).\n");
    }

    /* Optional: verify contents of d_b by copying back to host */
    float h_b[N];
    err = cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Copy back d_b to h_b failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Contents of d_b after correct copy:\n");
        for (int i = 0; i < N; ++i) {
            printf("%f ", h_b[i]);
        }
        printf("\n");
    }

    /* Clean up */
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
```