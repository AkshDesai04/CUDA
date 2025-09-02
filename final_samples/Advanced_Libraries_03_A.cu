```cpp
/*
 * Aim: Use `thrust::transform_reduce` to compute a dot product of two vectors in a single call.
 *
 * The idea behind this program is to illustrate how Thrust's `transform_reduce` can be used
 * to compute a dot product in one shot, without explicitly writing a loop or a separate
 * transformation kernel.  `transform_reduce` performs a transformation on each element of
 * an input range (here, the product of corresponding elements of two vectors) and then
 * reduces those transformed values using a binary operation (here, addition).
 *
 * We use two `thrust::device_vector`s to store the input data on the GPU.  A zip iterator
 * is created over the pair of vectors to expose each pair of elements as a single tuple.
 * The transformation is a lambda that multiplies the two components of the tuple.  The
 * reduction is performed with `thrust::plus<T>` to sum all products.  The identity value
 * for addition is `T(0)`.  The entire operation runs on the device thanks to the
 * `thrust::device` execution policy.
 *
 * The program demonstrates:
 *   1. Allocation and initialization of device vectors.
 *   2. Construction of a zip iterator.
 *   3. Use of a lambda for element‑wise multiplication.
 *   4. Single‑call `transform_reduce` to compute the dot product.
 *   5. Printing the result on the host.
 *
 * This code is self‑contained and can be compiled with `nvcc`:
 *     nvcc -O2 -std=c++14 -o dot_product dot_product.cu
 *
 * Running the executable prints the dot product of the two vectors.
 */

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>

int main()
{
    const std::size_t N = 1 << 20; // 1M elements
    using T = float;

    // Allocate and initialize host vectors
    thrust::host_vector<T> h_vec1(N);
    thrust::host_vector<T> h_vec2(N);
    for (std::size_t i = 0; i < N; ++i) {
        h_vec1[i] = static_cast<T>(i % 100);          // some pattern
        h_vec2[i] = static_cast<T>((i * 2) % 100);    // another pattern
    }

    // Transfer to device
    thrust::device_vector<T> d_vec1 = h_vec1;
    thrust::device_vector<T> d_vec2 = h_vec2;

    // Create zip iterator over the two device vectors
    auto first = thrust::make_zip_iterator(thrust::make_tuple(d_vec1.begin(), d_vec2.begin()));
    auto last  = thrust::make_zip_iterator(thrust::make_tuple(d_vec1.end(),   d_vec2.end()));

    // Compute dot product using transform_reduce
    // Transformation: multiply the two elements of each tuple
    // Reduction: add up all the products
    T dot_product = thrust::transform_reduce(
        thrust::device,                     // execution policy
        first,                              // start of zip iterator
        last,                               // end of zip iterator
        [] __device__ (const thrust::tuple<T,T>& t)
        {
            return thrust::get<0>(t) * thrust::get<1>(t);
        },
        static_cast<T>(0),                  // initial value for addition
        thrust::plus<T>()                   // reduction operation
    );

    std::cout << "Dot product of two vectors of size " << N << " is: " << dot_product << std::endl;

    return 0;
}
```