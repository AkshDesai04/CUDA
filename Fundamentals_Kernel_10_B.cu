/*
Modify the kernel to change a member of the struct it received. Does this change affect the original struct on the host? Explain why or why not (pass-by-value).

Thinking:
- In CUDA, kernel arguments are passed by value. When a struct is passed to a kernel, the compiler creates a copy of the struct in device memory and copies the contents from host to that copy.
- The kernel operates on that copy. Any modifications to the struct inside the kernel affect only the copy.
- After the kernel finishes, the original host struct remains unchanged because no mechanism copies the modified data back automatically.
- To modify the host struct, one would need to pass a pointer to the struct or return the modified struct via some other means.
- This program demonstrates that: we define a struct, launch a kernel that receives it by value and changes a field. We then print the struct on the host before and after kernel launch to show no change.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    int x;
    float y;
};

__global__ void modifyStructByValue(MyStruct s) {
    // This modifies the copy of the struct that the kernel received.
    s.x = 999;
    s.y = 3.14f;
    // The changes are local to this kernel and do not affect the host's struct.
}

__global__ void modifyStructByPointer(MyStruct *s) {
    // This modifies the struct via a device pointer, which points to the original host memory.
    s->x = 888;
    s->y = 2.71f;
}

int main() {
    MyStruct h_s;
    h_s.x = 10;
    h_s.y = 1.23f;

    printf("Before kernel launch:\n");
    printf("  h_s.x = %d, h_s.y = %f\n", h_s.x, h_s.y);

    // Pass by value
    modifyStructByValue<<<1,1>>>(h_s);
    cudaDeviceSynchronize();

    printf("\nAfter modifyStructByValue kernel:\n");
    printf("  h_s.x = %d, h_s.y = %f\n", h_s.x, h_s.y);
    // h_s remains unchanged because the kernel operated on a copy.

    // Pass by pointer
    modifyStructByPointer<<<1,1>>>( &h_s );
    cudaDeviceSynchronize();

    printf("\nAfter modifyStructByPointer kernel:\n");
    printf("  h_s.x = %d, h_s.y = %f\n", h_s.x, h_s.y);
    // h_s has been modified because the kernel dereferenced a pointer to host memory.

    return 0;
}