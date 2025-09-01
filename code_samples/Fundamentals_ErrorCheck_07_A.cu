/*
Aim of the program: Provide a list of 3 common synchronous CUDA API calls.

Thought process:
- The user explicitly asked for a list of three common synchronous CUDA API calls.
- According to the developer instruction, we must output a complete .cu file containing only the code, with no additional text.
- To respect the user’s request while still conforming to the instruction, the program will print a concise list of three commonly used synchronous CUDA API calls.
- The program will be written in standard C/C++ with CUDA extensions, and it will compile with nvcc.
- The main function will simply output the list using printf, as this is a synchronous operation and no device work is required.
- This satisfies both the user’s request for a list and the developer’s requirement to output a .cu file only.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Print a list of three common synchronous CUDA API calls
    printf("Common synchronous CUDA API calls:\n");
    printf("1. cudaMalloc\n");
    printf("2. cudaMemcpy\n");
    printf("3. cudaDeviceSynchronize\n");
    return 0;
}
