```cpp
/*
Aim: Pass the struct pointer, but forget to copy the data from host to device first.
What values does the kernel see when it reads from the struct's members?

Thinking:
1. We define a simple struct on the host with known values.
2. We allocate memory for this struct on the device but we deliberately do NOT
   copy the contents from the host struct to the device memory.
3. We launch a kernel that receives the device pointer to the struct and prints
   the values of its members.
4. Because the device memory was never initialized with the host data, the
   values read by the kernel are whatever happened to be in that region of
   device memory at the time of allocation.  In practice this is garbage /
   uninitialized data (it could even be zeros, but it is undefined).
5. We also add a second test where we actually copy the struct data to the
   device to confirm the difference.

Outcome:
- The kernel that uses the uninitialized device struct prints random values.
- The kernel that uses the correctly copied struct prints the expected values.

Note: On GPUs with UVA or unified virtual addressing, passing a host pointer
directly to the kernel would cause a fault; this code never does that – it
passes a device pointer but without initializing the contents.
*/

#include <cstdio>
#include <cuda_runtime.h>

struct MyStruct {
    int   a;
    float b;
};

__global__ void readStruct(const MyStruct* s, int idx) {
    // Each thread prints the contents of the struct
    printf("Thread %d sees struct: a = %d, b = %f\n", idx, s->a, s->b);
}

int main() {
    // Create and initialize host struct
    MyStruct h_s;
    h_s.a = 42;
    h_s.b = 3.1415f;

    // Allocate device memory for the struct but do NOT copy h_s to it
    MyStruct* d_s;
    cudaMalloc((void**)&d_s, sizeof(MyStruct));

    // Launch kernel with uninitialized struct data
    printf("=== Kernel with uninitialized device struct ===\n");
    readStruct<<<1, 4>>>(d_s, 0); // all threads read the same (garbage) struct
    cudaDeviceSynchronize();

    // Now copy the host struct to device
    cudaMemcpy(d_s, &h_s, sizeof(MyStruct), cudaMemcpyHostToDevice);

    // Launch kernel with initialized struct data
    printf("\n=== Kernel after copying struct to device ===\n");
    readStruct<<<1, 4>>>(d_s, 1);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_s);
    return 0;
}
```