```cuda
/*
 * Aim: Use memcmp to verify the entire block of memory at once instead of a loop.
 *
 * Thinking:
 * The goal of this program is to demonstrate a simple CUDA application that
 * performs a kernel operation on an array of integers, copies the result
 * back to the host, and then verifies the correctness of the entire memory
 * block in one go using memcmp. Instead of iterating element‑by‑element
 * to compare the expected output with the actual output, we use the C
 * library function memcmp which compares raw bytes.  This is efficient
 * because it can be highly optimised by the host CPU, and it simplifies
 * the verification logic.
 *
 * The program flow is:
 * 1. Allocate a host array h_input of N integers and fill it with sequential
 *    values (0, 1, 2, …).
 * 2. Allocate a host array h_expected that will hold the expected result
 *    after the kernel operation (increment each element by 1).
 * 3. Allocate device memory d_data and copy h_input to the device.
 * 4. Launch a simple kernel that adds 1 to each element of the array.
 * 5. Copy the device result back to h_output on the host.
 * 6. Compute the expected result in h_expected.
 * 7. Use memcmp to compare h_expected and h_output in one call.
 * 8. Report success or failure.
 *
 * This demonstrates how to use memcmp for bulk memory verification in a
 * CUDA context, avoiding an explicit element‑wise loop on the host side.
 */

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void increment_kernel(int *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_data[idx] += 1;
}

int main()
{
    const int N = 1 << 20; // 1M integers
    const size_t size_bytes = N * sizeof(int);

    // Allocate host memory
    int *h_input = (int*)malloc(size_bytes);
    int *h_output = (int*)malloc(size_bytes);
    int *h_expected = (int*)malloc(size_bytes);
    if (!h_input || !h_output || !h_expected) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i)
        h_input[i] = i;

    // Allocate device memory
    int *d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, size_bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_data, h_input, size_bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocks, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_data, size_bytes, cudaMemcpyDeviceToHost));

    // Compute expected result
    for (int i = 0; i < N; ++i)
        h_expected[i] = h_input[i] + 1;

    // Verify using memcmp
    if (memcmp(h_expected, h_output, size_bytes) == 0) {
        printf("Success: Device result matches expected output.\n");
    } else {
        printf("Error: Device result does not match expected output.\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_expected);
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```