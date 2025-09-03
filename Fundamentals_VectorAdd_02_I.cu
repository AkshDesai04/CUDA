```cpp
/* 
Aim: Create a main function that accepts N as a command-line argument. 
The goal of this program is simply to demonstrate how to read a single
integer from the command line and use it in a CUDA-enabled application. 
We parse the first argument (after the program name), convert it to an
integer, and print it out. If the argument is missing or invalid, we
display an error message and exit. This is a minimal example; in a
real CUDA application we might use N to determine the size of a data
array or the number of threads, but that is beyond the scope of this
simple demonstration. The code is written in C/C++ with CUDA extensions
and can be compiled with nvcc. */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* Convert the command-line argument to an integer */
    char *endptr;
    long n_long = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || n_long <= 0) {
        fprintf(stderr, "Invalid integer value for N: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    int N = (int)n_long;

    printf("Received N = %d\n", N);

    /* Here you could launch a CUDA kernel that uses N, e.g. */
    /* Example: Allocate device memory and launch kernel */
    /* (Omitted for brevity) */

    return EXIT_SUCCESS;
}
```