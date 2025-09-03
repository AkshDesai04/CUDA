/*
Deliberately corrupt one byte of the data on the host before verification to ensure your verification code works.

My thinking:
1. The goal is to demonstrate that the verification routine can detect a corrupted data element.
2. I'll allocate an array of integers on the host, initialize it with a predictable pattern.
3. The CUDA kernel will simply increment each element by 1.
4. After copying the result back to the host, I will deliberately corrupt a single byte of one integer in the host array (e.g., flip the least significant byte of element 42).
5. The verification routine will compare each element against the expected value (original + 1). The corrupted element will fail the check, and the program will report a mismatch.
6. I include a helper macro for CUDA error checking and keep the code self-contained and compilable.
7. The program will print a success message if all elements match or an error message indicating the first mismatch.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstdint>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                      \
              #call, __FILE__, __LINE__, cudaGetErrorString(err));          \
      exit(EXIT_FAILURE);                                                   \
    }                                                                        \
  } while (0)

__global__ void increment_kernel(int *d_arr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d_arr[idx] += 1;
  }
}

int main(void) {
  const int N = 256;
  size_t bytes = N * sizeof(int);

  // Host allocation
  int *h_input = (int *)malloc(bytes);
  int *h_output = (int *)malloc(bytes);
  if (!h_input || !h_output) {
    fprintf(stderr, "Failed to allocate host memory.\n");
    return EXIT_FAILURE;
  }

  // Initialize host input with known pattern
  for (int i = 0; i < N; ++i) {
    h_input[i] = i * 10;   // 0,10,20,...
  }

  // Device allocation
  int *d_arr = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_arr, bytes));

  // Copy input to device
  CUDA_CHECK(cudaMemcpy(d_arr, h_input, bytes, cudaMemcpyHostToDevice));

  // Launch kernel
  int threadsPerBlock = 128;
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  increment_kernel<<<blocks, threadsPerBlock>>>(d_arr, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(h_output, d_arr, bytes, cudaMemcpyDeviceToHost));

  // Deliberately corrupt one byte of the host output
  // Choose element 42 and flip its least significant byte
  int corrupt_index = 42;
  unsigned char *byte_ptr = reinterpret_cast<unsigned char *>(&h_output[corrupt_index]);
  *byte_ptr ^= 0xFF;  // flip all bits in the first byte

  // Verification
  bool success = true;
  for (int i = 0; i < N; ++i) {
    int expected = h_input[i] + 1;
    if (h_output[i] != expected) {
      fprintf(stderr,
              "Verification failed at index %d: expected %d, got %d\n",
              i, expected, h_output[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("All values verified successfully.\n");
  } else {
    printf("Verification failed due to corrupted data.\n");
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_arr));
  free(h_input);
  free(h_output);

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
