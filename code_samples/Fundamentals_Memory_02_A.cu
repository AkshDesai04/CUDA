/*
Write a loop that iterates through the arrays and prints an error message if a mismatch is found, then breaks.
[Thinking: The goal is to demonstrate a simple element‑by‑element comparison between two arrays in CUDA C.  Because the requirement is just a loop that checks for mismatches and prints an error before breaking, we can implement this entirely on the host.  A small example will use two integer arrays of the same size, deliberately introduce a mismatch, and then run a for loop to find the first mismatch, print an informative message (including the index and the differing values), and then exit the loop with a break.  The code is kept minimal and uses standard CUDA C constructs so that it can be compiled with nvcc.  No device kernels are necessary for this logic, but the file will have a .cu extension to satisfy the request.  The program also prints a success message if no mismatches are found, to make the output clear when the arrays are identical.  The code will be self‑contained and compilable on any system with CUDA installed.*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    const int N = 10;
    int hostArray1[N];
    int hostArray2[N];

    // Initialize both arrays with the same values
    for (int i = 0; i < N; ++i)
    {
        hostArray1[i] = i * 2;
        hostArray2[i] = i * 2;
    }

    // Intentionally create a mismatch at index 5
    hostArray2[5] = 999;

    // Loop through arrays to find first mismatch
    int mismatchFound = 0;
    for (int i = 0; i < N; ++i)
    {
        if (hostArray1[i] != hostArray2[i])
        {
            printf("Error: Mismatch found at index %d. "
                   "hostArray1[%d] = %d, hostArray2[%d] = %d\n",
                   i, i, hostArray1[i], i, hostArray2[i]);
            mismatchFound = 1;
            break;
        }
    }

    if (!mismatchFound)
    {
        printf("No mismatches found. Arrays are identical.\n");
    }

    return 0;
}
