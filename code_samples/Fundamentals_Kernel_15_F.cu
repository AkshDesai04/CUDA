/*
Aim of the program: Launch the hanging kernel. Does the `main` function ever exit? Why or why not?

Thinking:
1. The goal is to create a CUDA program that launches a kernel which never terminates (an infinite loop). 
2. In CUDA, kernel launches are asynchronous; however, the CUDA runtime ensures that the host process does not exit until all outstanding device operations have completed or are aborted.
3. Since the kernel we launch contains an infinite loop (while(true)), it will never finish. The runtime will block the host thread when it tries to destroy the context at program termination, effectively preventing `main` from ever exiting.
4. The program will hang, demonstrating that the main function does not exit until the kernel finishes. 
5. The code will launch the kernel, and then simply return. No explicit cudaDeviceSynchronize() or other sync primitives are used; the hang will be observed at program exit.
6. The kernel will run on one block with one thread for simplicity, and it will simply spin forever.

The program below reflects this reasoning. It can be compiled with `nvcc` and executed; it will hang and never return to the shell.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// A kernel that spins forever, never returning
__global__ void hangingKernel() {
    // Each thread enters an infinite loop
    while (true) {
        // Do nothing
    }
}

int main() {
    // Launch the hanging kernel with 1 block of 1 thread
    hangingKernel<<<1, 1>>>();

    // No cudaDeviceSynchronize() or other sync; we simply return
    // The CUDA runtime will block on exit until the kernel finishes,
    // which in this case never happens, so main never exits.
    printf("Kernel launched, main is about to exit (but will block due to hanging kernel).\n");
    return 0;
}
