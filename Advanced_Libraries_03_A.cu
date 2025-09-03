/*
Aim: Use `thrust::transform_reduce` to compute a dot product of two vectors in a single call.

Thinking:
- The program demonstrates how to compute the dot product of two numeric vectors efficiently on the GPU using the Thrust library.
- We will create two device vectors of floating-point numbers, populate them with example values (e.g., sequential data), and then use thrust::transform_reduce to perform the dot product.
- transform_reduce takes an input range and a second input range, applies a binary operation (here we use thrust::multiplies to multiply corresponding elements), then reduces the result with a binary operation (thrust::plus to sum).
- The template signature: transform_reduce(ExecutionPolicy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryOp op, T init, BinaryOp binary_op)
  In our case, op will be thrust::multiplies which multiplies elements from the two ranges.
- The code uses thrust::device execution policy for GPU execution. It prints the resulting dot product to standard output.
- Error handling is minimal; the program focuses on illustrating the use of transform_reduce.
*/

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>

int main()
{
    const std::size_t N = 1 << 20; // Example size: 1 million elements

    // Create two device vectors and fill them with sequential values
    thrust::device_vector<float> a(N);
    thrust::device_vector<float> b(N);

    // Fill vector a with 1.0, 2.0, 3.0, ...
    thrust::sequence(a.begin(), a.end(), 1.0f);
    // Fill vector b with 2.0, 4.0, 6.0, ...
    thrust::sequence(b.begin(), b.end(), 2.0f);

    // Compute dot product using thrust::transform_reduce
    // Multiply corresponding elements and sum the results
    float dot_product = thrust::transform_reduce(
        thrust::device,                // Execution policy
        a.begin(),                     // First range beginning
        a.end(),                       // First range end
        b.begin(),                     // Second range beginning
        thrust::multiplies<float>(),   // Binary operation: multiply
        0.0f,                          // Initial value for reduction
        thrust::plus<float>()          // Reduction operation: sum
    );

    std::cout << "Dot product: " << dot_product << std::endl;

    return 0;
}
