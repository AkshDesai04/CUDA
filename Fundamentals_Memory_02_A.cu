/*
Write a loop that iterates through the arrays and prints an error message if a mismatch is found, then breaks.

Thinking:
The goal is to create a CUDA C (.cu) program that demonstrates how to iterate over two arrays, compare each corresponding element, and if a mismatch is found, print an informative error message and exit the loop. The program will be simple enough to run on the host side, but it will still be compiled as a CUDA source file to adhere to the requirement of a .cu file. The arrays are allocated on the host, filled with test data, and a deliberate mismatch is introduced to showcase the detection logic. The loop will iterate over the arrays, compare values, and upon encountering a mismatch, use `printf` to output the index and differing values, then `break` out of the loop. After the loop, the program will print a success message if no mismatches were found. This code is straightforward, portable, and easily extendable for more complex CUDA operations if desired.
*/

#include <stdio.h>

int main(void) {
    const int N = 10;
    int a[N], b[N];
    int i;

    /* Initialize arrays with matching values */
    for (i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    /* Introduce a deliberate mismatch for demonstration */
    b[5] = 999;  /* This will not match a[5] */

    /* Loop through arrays, compare, and break on mismatch */
    for (i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch found at index %d: a[%d] = %d, b[%d] = %d\n",
                   i, i, a[i], i, b[i]);
            break;
        }
    }

    if (i == N) {
        printf("All array elements match.\n");
    }

    return 0;
}
