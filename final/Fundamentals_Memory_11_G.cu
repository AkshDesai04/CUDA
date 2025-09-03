```cpp
/*
Aim of the program: Use `cudaGetSymbolAddress` to get the device address of the constant variable.

Thinking process:
1. CUDA has a special memory region called constant memory, which is read‑only for kernels but can be written by the host using `cudaMemcpyToSymbol` or `cudaMemcpyFromSymbol`. The symbol representing a constant variable is available on the host side as an external reference.

2. The CUDA runtime provides `cudaGetSymbolAddress(void **devPtr, const void *symbol)` which returns the device pointer corresponding to a symbol name (e.g., a global or constant variable). This pointer can be used in subsequent CUDA API calls such as `cudaMemcpy`, or passed directly to kernels as a parameter.

3. In the example below, we:
   - Declare a constant integer `d_constVar` in constant memory.
   - Initialize it on the host with `cudaMemcpyToSymbol`.
   - Retrieve its device address via `cudaGetSymbolAddress`.
   - Use that address to copy the value back to the host using `cudaMemcpy` (to demonstrate that the address points to the correct memory).
   - Launch a simple kernel that receives the address of the constant variable as an argument and prints the value it reads, proving that the address is valid inside device code.

4. We include basic error checking with a helper macro `CHECK_CUDA` to make debugging easier.

5. The program is written in standard CUDA C (extension .cu) and can be compiled with `nvcc`.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Check CUDA errors */
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

/* Declare a constant variable in device constant memory */
__constant__ int d_constVar;

/* Simple kernel that reads from the constant variable via its address */
__global__ void kernelReadConst(const int *constPtr)
{
    // Only one thread prints for demonstration
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Kernel read value: %d\n", *constPtr);
    }
}

int main(void)
{
    /* Host value to copy into device constant memory */
    const int h_val = 42;

    /* Copy value to constant memory */
    CHECK_CUDA(cudaMemcpyToSymbol(d_constVar, &h_val, sizeof(int), 0, cudaMemcpyHostToDevice));

    /* Retrieve device address of the constant variable */
    void *devAddr = nullptr;
    CHECK_CUDA(cudaGetSymbolAddress(&devAddr, d_constVar));

    printf("Device address of d_constVar: %p\n", devAddr);

    /* Verify that we can read back the value using cudaMemcpy and the address */
    int h_copy = 0;
    CHECK_CUDA(cudaMemcpy(&h_copy, devAddr, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Value copied back from constant memory: %d\n", h_copy);

    /* Launch kernel, passing the device address */
    kernelReadConst<<<1, 1>>>(reinterpret_cast<const int *>(devAddr));
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}
```