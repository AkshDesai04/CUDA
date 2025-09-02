/*
Aim: Create a macro `CHECK_KERNEL(kernel_launch)` that expands to `{ kernel_launch; checkKernelLaunch(...); }`.

Thinking:
- The macro should accept a kernel launch expression (e.g., `myKernel<<<grid, block>>>(args);`).
- After launching, we want to immediately check for launch errors using a helper function `checkKernelLaunch`.
- The helper function will call `cudaGetLastError()` to catch any errors that occurred during the kernel launch.
- To provide useful debugging information, the helper will also print the file name and line number where the macro was invoked. These are passed via the special macros `__FILE__` and `__LINE__`.
- The macro should expand to a block so that it can be used safely in an expression context.
- Implementation details:
  1. Include `<cuda_runtime.h>` for CUDA runtime API.
  2. Implement `checkKernelLaunch(const char* file, int line)` which checks `cudaGetLastError()` and prints an error message if needed, then exits.
  3. Define the macro using the `do { ... } while(0)` idiom to ensure safe single-statement usage (though the prompt explicitly wants `{ kernel_launch; checkKernelLaunch(...); }`, we'll keep it as a block without `do-while` to match the requirement exactly, but still safe enough).
  4. Provide a small example kernel and a `main` function that demonstrates usage of the macro. This is optional but helps illustrate how the macro works.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/* Helper function to check kernel launch errors */
void checkKernelLaunch(const char *file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA error at %s:%d: %s\n",
                file,
                line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Macro that launches a kernel and checks for launch errors */
#define CHECK_KERNEL(kernel_launch) \
    { kernel_launch; checkKernelLaunch(__FILE__, __LINE__); }

/* Example kernel */
__global__ void exampleKernel(int *data, int val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = val;
}

int main(void)
{
    const int N = 256;
    int *d_data;

    /* Allocate device memory */
    cudaMalloc((void **)&d_data, N * sizeof(int));

    /* Launch the kernel with the macro */
    CHECK_KERNEL(exampleKernel<<<(N + 255) / 256, 256>>>(d_data, 42));

    /* Free resources */
    cudaFree(d_data);

    printf("Kernel launched successfully.\n");
    return 0;
}
