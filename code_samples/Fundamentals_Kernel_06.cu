/*
Write a kernel that accepts an integer pointer `int* data` as an argument. The host code should allocate a single integer on the GPU (`cudaMalloc`), launch the kernel with one thread, and pass the pointer. The kernel should write the value `1337` to the memory location pointed to by `data`. Finally, the host should copy the value back (`cudaMemcpy`) and print it to verify.

**Thinking Process:**
1. **Kernel Design**: The kernel is trivial: it receives a pointer to an int and writes 1337 to that address. Since only one thread will run, we can directly dereference the pointer.
2. **Memory Allocation**: On the host, we need to allocate space for a single integer on the GPU using `cudaMalloc`. This returns a device pointer that we will pass to the kernel.
3. **Kernel Launch**: We launch the kernel with a single block and a single thread (`<<<1,1>>>`) because we only need one thread to write the value.
4. **Synchronization**: After launching the kernel, we should call `cudaDeviceSynchronize()` to ensure the kernel has finished executing before we copy data back.
5. **Copy Back and Print**: Use `cudaMemcpy` to copy the device integer back to a host integer variable. Print the value to verify it is 1337.
6. **Cleanup**: Free the device memory with `cudaFree`.

Edge cases are minimal here: we only allocate a single int, so no need for boundary checks. Basic error checking could be added but is omitted for brevity.

The final code below implements this logic.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes 1337 to the integer pointed to by `data`
__global__ void writeKernel(int *data) {
    // Since there is only one thread, we directly write the value
    *data = 1337;
}

int main() {
    int *d_val = NULL;   // Device pointer
    int h_val = 0;       // Host variable to hold the result

    // Allocate memory for one integer on the GPU
    cudaError_t err = cudaMalloc((void **)&d_val, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch the kernel with one block of one thread
    writeKernel<<<1, 1>>>(d_val);

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_val);
        return -1;
    }

    // Copy the result back to the host
    err = cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_val);
        return -1;
    }

    // Print the value to verify
    printf("Value read from device: %d\n", h_val);

    // Clean up
    cudaFree(d_val);

    return 0;
}
