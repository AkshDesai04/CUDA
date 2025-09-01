/*
Modify the kernel to change a member of the struct it received.
Does this change affect the original struct on the host? Explain why or why not (pass-by-value).

Thinking:
- The goal is to demonstrate that when a struct is passed to a CUDA kernel by value,
  the kernel receives a copy of that struct. Any modifications inside the kernel
  affect only the copy, not the original object in host memory.
- We define a simple struct `MyStruct` with two members.
- The kernel `modifyStruct` takes a `MyStruct` parameter by value and changes its members.
- On the host, we create an instance of `MyStruct`, print its values, invoke the kernel,
  and then print the values again to show that they have not changed.
- Since CUDA automatically copies kernel arguments from host to device for by-value parameters
  and does not copy them back, the host version remains untouched.
- We also use `cudaMemcpy` to copy the struct to device memory, but that copy is unnecessary
  for by-value passing; however, it helps illustrate that we could still copy data back
  manually if desired. In this example, we rely on the implicit copy performed by the
  kernel launch and show that the host data stays the same.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err)                                 \
    do {                                                \
        cudaError_t err__ = (err);                      \
        if (err__ != cudaSuccess) {                     \
            std::cerr << "CUDA error: "                 \
                      << cudaGetErrorString(err__)      \
                      << " at line " << __LINE__        \
                      << " in file " << __FILE__ << std::endl; \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)

// Define a simple struct
struct MyStruct {
    int   a;
    float b;
};

// Kernel that receives the struct by value and modifies its members
__global__ void modifyStruct(MyStruct s) {
    // Modify the copy of the struct
    s.a = 42;
    s.b = 3.14f;

    // Print inside kernel to show the values being modified
    printf("Kernel sees struct: a = %d, b = %f\n", s.a, s.b);
}

int main() {
    // Create an instance of MyStruct on host
    MyStruct h_struct;
    h_struct.a = 1;
    h_struct.b = 2.0f;

    std::cout << "Before kernel launch:\n";
    std::cout << "Host struct: a = " << h_struct.a
              << ", b = " << h_struct.b << "\n";

    // Launch kernel with the struct passed by value
    modifyStruct<<<1, 1>>>(h_struct);
    CUDA_CHECK(cudaDeviceSynchronize());

    // After kernel, host struct should be unchanged because pass-by-value
    std::cout << "After kernel launch:\n";
    std::cout << "Host struct: a = " << h_struct.a
              << ", b = " << h_struct.b << "\n";

    // Clean up
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
