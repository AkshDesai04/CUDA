```cuda
/*
Aim: Write a program that deliberately triggers a cudaErrorInvalidValue by passing a nullptr as the destination for cudaMemcpy. Use the CHECK() macro to catch and report the error cleanly.

Thinking:
- The CHECK macro should wrap a CUDA API call, evaluate its return status, and if it indicates an error, print the error string and exit the program. This keeps error handling concise.
- To trigger cudaErrorInvalidValue, we will call cudaMemcpy with a null destination pointer. The source can be a small buffer allocated on the host; the size and kind can be anything valid, but the null dest will make the API fail.
- We need to include the necessary headers: <cstdio> for printf, <cstdlib> for exit, and <cuda_runtime.h> for CUDA runtime API.
- The program will allocate a small host buffer, then call cudaMemcpy(host_ptr, nullptr, 4, cudaMemcpyHostToDevice). This should produce an error. The CHECK macro will catch this and print a message.
- We also ensure we free any allocated host memory before exiting (although the program will abort upon error).
- The code will be a single .cu file ready to compile with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

int main(void) {
    const int N = 4;
    int *h_ptr = (int*)malloc(N * sizeof(int));
    if (!h_ptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_ptr[i] = i;
    }

    // Intentionally pass nullptr as destination to trigger cudaErrorInvalidValue
    CHECK(cudaMemcpy(nullptr, h_ptr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Clean up (will not be reached due to CHECK aborting)
    free(h_ptr);
    return 0;
}
```