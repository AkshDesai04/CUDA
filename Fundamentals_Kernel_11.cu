/*
Pass a pointer to a struct to a kernel. The host should allocate the struct on both host and device. Initialize it on the host, copy to device. The kernel should modify a member of the struct. The host should copy it back and verify the modification.

To achieve this, we define a simple struct with a few members. The host allocates an instance of this struct and sets its fields. A device copy of the struct is allocated using cudaMalloc, and the contents are copied from host to device with cudaMemcpy. The kernel receives a pointer to the struct on device memory and updates its fields. After kernel execution, the struct is copied back to the host and its updated values are printed to verify the modification. Error checking is performed after each CUDA call to catch any issues. The program demonstrates pointer passing of a struct to a kernel and the correct usage of cudaMemcpy for device-to-host and host-to-device transfers. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    int   a;
    float b;
    char  c;
};

// Macro for error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that modifies the struct
__global__ void modifyStruct(MyStruct *s) {
    // Only one thread is launched; no need to check threadIdx
    s->a = 42;
    s->b = 2.718f;
    // Leaving 'c' unchanged to verify that other members are unaffected
}

int main(void) {
    // Host struct
    MyStruct h_struct;
    h_struct.a = 5;
    h_struct.b = 3.14f;
    h_struct.c = 'x';

    printf("Before CUDA:\n");
    printf("  h_struct.a = %d\n", h_struct.a);
    printf("  h_struct.b = %f\n", h_struct.b);
    printf("  h_struct.c = %c\n", h_struct.c);

    // Device struct pointer
    MyStruct *d_struct = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_struct, sizeof(MyStruct)));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct), cudaMemcpyHostToDevice));

    // Launch kernel (1 block, 1 thread)
    modifyStruct<<<1, 1>>>(d_struct);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy device back to host
    CHECK_CUDA(cudaMemcpy(&h_struct, d_struct, sizeof(MyStruct), cudaMemcpyDeviceToHost));

    printf("\nAfter CUDA:\n");
    printf("  h_struct.a = %d\n", h_struct.a);
    printf("  h_struct.b = %f\n", h_struct.b);
    printf("  h_struct.c = %c\n", h_struct.c);

    // Verify the modifications
    if (h_struct.a == 42 && h_struct.b == 2.718f) {
        printf("\nVerification succeeded: struct was modified correctly.\n");
    } else {
        printf("\nVerification failed: struct was not modified as expected.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_struct));

    return 0;
}
