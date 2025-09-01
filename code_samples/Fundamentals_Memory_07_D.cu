```cpp
/*
Aim of the program (word for word):
"If you print the value of a device pointer (`printf("%p", d_ptr);`), it prints an address. Explain that this address is only valid from the GPU's perspective."

Thinking:
1. The program allocates memory on the GPU using `cudaMalloc`.  
2. It then launches a kernel that receives the device pointer `d_ptr` as an argument.  
3. Inside the kernel, `printf` is used to print the address stored in `d_ptr`.  
   Because the kernel runs on the GPU, this address is interpreted in the GPU's
   virtual address space and is therefore meaningful only to the GPU.  
4. After the kernel finishes, the host program also prints the same device
   pointer value.  The numeric value is the same, but the host cannot dereference
   it – attempting to read from that address will result in an invalid
   memory access.  
5. For comparison, the host also prints the address of a host-allocated array
   to show that host addresses and device addresses are in completely
   separate address spaces.  
6. The output demonstrates that the address printed by `printf` in a kernel
   is valid only on the GPU side, while the host side cannot use it to
   access the GPU memory.

This simple program illustrates the key concept that device pointers are
GPU-specific addresses and must not be treated as regular pointers on the
CPU side. The code is written in CUDA C and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printDevicePointer(int *d_ptr, int idx)
{
    // Print the address stored in d_ptr from the GPU perspective.
    printf("Kernel (thread %d): Device pointer address = %p\n", idx, (void *)d_ptr);
}

int main()
{
    // Allocate an array on the device.
    int *d_arr = NULL;
    size_t N = 10;
    size_t bytes = N * sizeof(int);
    cudaError_t err = cudaMalloc((void **)&d_arr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Allocate an array on the host for comparison.
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "malloc failed\n");
        cudaFree(d_arr);
        return -1;
    }

    // Launch kernel to print the device pointer address.
    printDevicePointer<<<1, 1>>>(d_arr, 0);
    cudaDeviceSynchronize();

    // Print the device pointer from the host side.
    printf("Host: Device pointer value = %p\n", (void *)d_arr);

    // Print the host pointer value.
    printf("Host: Host array pointer value = %p\n", (void *)h_arr);

    // Clean up.
    free(h_arr);
    cudaFree(d_arr);

    return 0;
}
```