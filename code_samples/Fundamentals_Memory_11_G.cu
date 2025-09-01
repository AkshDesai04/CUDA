/*
Aim: Use `cudaGetSymbolAddress` to get the device address of the constant variable.

Thinking:
1. In CUDA, constant memory is a special read‑only region on the device.  
   The compiler keeps a symbol for each constant variable, and the host can
   query the device address of that symbol using `cudaGetSymbolAddress`.  
2. After obtaining the device address, the host can copy data from or to
   that address using `cudaMemcpy` (or `cudaMemcpyFromSymbol`).  
3. The program will:
   - Define a constant integer `d_constVar` initialized to 42.
   - Launch a simple kernel that writes this constant to a device array.
   - Use `cudaGetSymbolAddress` to retrieve the device pointer to `d_constVar`.
   - Verify the address by copying the value from that address back to host
     memory and printing it.
   - Clean up device memory.
4. Error checking is added for all CUDA API calls to ensure proper diagnostics.
5. The code is self‑contained and can be compiled with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Constant variable stored in device constant memory
__constant__ int d_constVar = 42;

// Simple kernel that writes the constant value to an array
__global__ void kernel(int *out) {
    int idx = threadIdx.x;
    out[idx] = d_constVar;
}

int main() {
    const int N = 10;
    int h_out[N];
    int *d_out = nullptr;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel
    kernel<<<1, N>>>(d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    printf("Kernel output (should be all 42s):\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Obtain the device address of the constant variable
    void *d_addr = nullptr;
    err = cudaGetSymbolAddress(&d_addr, d_constVar);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetSymbolAddress failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }
    printf("Device address of d_constVar: %p\n", d_addr);

    // Copy the constant value from device to host using the obtained address
    int h_val_from_addr = 0;
    err = cudaMemcpy(&h_val_from_addr, d_addr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy from symbol address failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }
    printf("Value read from constant memory via address: %d\n", h_val_from_addr);

    // Alternatively, use cudaMemcpyFromSymbol directly (for demonstration)
    int h_val_from_symbol = 0;
    err = cudaMemcpyFromSymbol(&h_val_from_symbol, d_constVar, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyFromSymbol failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }
    printf("Value read from constant memory via cudaMemcpyFromSymbol: %d\n", h_val_from_symbol);

    // Clean up
    cudaFree(d_out);
    return 0;
}
