/*
Create a C/C++ preprocessor macro `CHECK(call)` that wraps a CUDA API call. The macro should check the returned `cudaError_t` value. If it's not `cudaSuccess`, it should print the error code, the error string from `cudaGetErrorString`, the file name (`__FILE__`), and the line number (`__LINE__`), and then exit the program.

Thinking process:
1. The macro must be safe for any expression, so we wrap it in a `do { ... } while(0)` block.
2. We capture the result of the call into a local variable of type `cudaError_t`.
3. If the result is not `cudaSuccess`, we use `fprintf(stderr, ...)` to print the file and line, the error string via `cudaGetErrorString`, and the numeric error code.
4. After printing, we call `exit(EXIT_FAILURE)` to terminate the program.
5. The macro is defined in a way that it can be used anywhere after including `<cuda_runtime.h>` and standard headers.
6. To demonstrate usage, a minimal `main` that calls a dummy CUDA API (e.g., `cudaSetDevice(0)`) wrapped by `CHECK` is included. This will compile and run on a system with CUDA installed.

The final .cu file contains the macro definition, necessary includes, and a sample `main`.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/* Macro to check CUDA API call results */
#define CHECK(call) do {                                            \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

int main(void)
{
    /* Example usage of the CHECK macro */
    CHECK(cudaSetDevice(0));  /* Attempt to set CUDA device 0 */

    /* Dummy kernel launch (no actual kernel defined) */
    /* To keep this example minimal, we skip launching a kernel. */

    printf("CUDA check passed, program completed successfully.\n");
    return 0;
}