```cuda
/*
Aim of the program: Use `sizeof()` on the host and device to print the size of the struct and verify they are the same.

Thoughts:
1. Define a simple struct with a few members of different types.
2. On the host, use `sizeof(MyStruct)` and print it using std::cout.
3. On the device, create a kernel that prints `sizeof(MyStruct)` using device `printf`. 
   This requires a compute capability of at least 2.0, so the code should be compiled
   with an appropriate architecture flag (e.g. -arch=sm_20).
4. Launch the kernel from the host, then synchronize and exit.
5. No dynamic memory or complex error handling is needed; the focus is on demonstrating
   that `sizeof()` returns the same value on both host and device.
6. The program will be self‑contained in a single .cu file, with all includes and
   using statements necessary for compilation.
*/

#include <cstdio>
#include <iostream>

// Define a struct with diverse members
struct MyStruct {
    int    a;
    double b;
    char   c;
};

__global__ void printSizeKernel()
{
    // Print size of the struct from the device side
    printf("Size of struct on device: %zu\n", sizeof(MyStruct));
}

int main()
{
    // Print size of the struct on the host side
    std::cout << "Size of struct on host: " << sizeof(MyStruct) << std::endl;

    // Launch kernel to print size on device
    printSizeKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```