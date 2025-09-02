```cpp
/*
Aim of the program: Write a __host__ function that takes an int* and dereferences it. Call it with a device pointer. Observe the segfault.

Thinking:
1. CUDA device memory cannot be directly accessed by host code. Trying to dereference a device pointer on the host will lead to a segmentation fault or illegal memory access.
2. To demonstrate this, we allocate an integer on the device using cudaMalloc.
3. We write a small kernel that writes a value to that device integer so that the pointer is valid and points to a real allocation.
4. We then call a __host__ function that simply tries to read the integer through the pointer using the dereference operator.
5. When the host function attempts to read the device memory, the CUDA runtime will not translate the address; the CPU will try to read from an invalid host address, causing a segfault.
6. The program is intentionally simple; no error checking is performed for brevity, because the focus is on the segfault behavior.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Host function that dereferences an int* and prints its value
__host__ void hostDereference(int *ptr) {
    // Attempt to dereference the device pointer on the host
    printf("Attempting to dereference device pointer on host. Value: %d\n", *ptr);
}

// Kernel to set a value on the device
__global__ void setValue(int *ptr, int val) {
    *ptr = val;
}

int main() {
    int *d_ptr = NULL;
    int val = 123;  // Value to store on device

    // Allocate device memory for an integer
    cudaMalloc((void**)&d_ptr, sizeof(int));

    // Launch kernel to write the value to device memory
    setValue<<<1, 1>>>(d_ptr, val);
    cudaDeviceSynchronize();

    // Call the host function with the device pointer (expected to segfault)
    hostDereference(d_ptr);

    // Cleanup (this line may not be reached if segfault occurs)
    cudaFree(d_ptr);

    return 0;
}
```