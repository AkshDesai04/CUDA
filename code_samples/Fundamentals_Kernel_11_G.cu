/*
Pass the struct pointer, have the kernel modify it, but forget to copy the data back from device to host. Does the verification on the host pass?

The goal of this program is to demonstrate that if you modify a struct on the device via a kernel, but never copy the updated data back to the host, the host will still see the original values. Therefore, any verification performed on the host that checks for the updated values will fail.

Steps in the program:
1. Define a simple struct with an int and a float.
2. Allocate an instance of the struct on the host and initialize it.
3. Allocate memory for the struct on the device and copy the host struct to the device.
4. Launch a kernel that writes new values into the struct on the device.
5. Do NOT copy the struct back from device to host.
6. Verify the values on the host; the check should fail because the host still holds the original values.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Simple struct with two members
typedef struct {
    int   a;
    float b;
} MyStruct;

// Kernel that modifies the struct on the device
__global__ void modifyStruct(MyStruct *s)
{
    // Simple modification: set new values
    s->a = 42;
    s->b = 3.14f;
}

int main()
{
    // Host struct
    MyStruct h_s;
    h_s.a = 0;
    h_s.b = 0.0f;

    // Device pointer
    MyStruct *d_s = NULL;
    cudaError_t err;

    // Allocate memory on the device
    err = cudaMalloc((void **)&d_s, sizeof(MyStruct));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy host struct to device
    err = cudaMemcpy(d_s, &h_s, sizeof(MyStruct), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_s);
        return -1;
    }

    // Launch kernel
    modifyStruct<<<1, 1>>>(d_s);

    // Synchronize to ensure kernel has finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_s);
        return -1;
    }

    // IMPORTANT: We intentionally skip copying the struct back from device to host.

    // Verification on host
    printf("Host struct after kernel launch (without memcpy back):\n");
    printf("  a = %d\n", h_s.a);
    printf("  b = %f\n", h_s.b);

    if (h_s.a == 42 && fabsf(h_s.b - 3.14f) < 1e-5f) {
        printf("PASS: Host struct has been updated.\n");
    } else {
        printf("FAIL: Host struct has NOT been updated.\n");
    }

    // Clean up
    cudaFree(d_s);

    return 0;
}
