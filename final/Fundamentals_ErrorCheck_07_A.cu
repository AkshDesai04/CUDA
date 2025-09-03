/*
Provide a list of 3 common synchronous CUDA API calls.

Thinking:
The user explicitly requests a list of three common synchronous CUDA API calls. According to the developer instructions, we must produce a CUDA C (.cu) file containing the entire code, preceded by a multiline comment that states the aim of the program exactly as the user provided, and also includes all of our internal reasoning. After the comment block, the actual C/CUDA code follows. Since the user only wants a list, we can create a minimal host program that prints these calls to stdout. The program will include the standard CUDA headers and a simple main function that prints the names of the calls. This satisfies both the user’s request for the list and the developer’s requirement for a .cu file output.

Three common synchronous CUDA API calls are:
1. cudaMemcpy
2. cudaMemset
3. cudaDeviceSynchronize

We'll implement a small program that outputs these three calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    const char* calls[] = {
        "cudaMemcpy",
        "cudaMemset",
        "cudaDeviceSynchronize"
    };

    printf("Common synchronous CUDA API calls:\n");
    for (int i = 0; i < 3; ++i) {
        printf("%d. %s\n", i + 1, calls[i]);
    }

    return 0;
}
