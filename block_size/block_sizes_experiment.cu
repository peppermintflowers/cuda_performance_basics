#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

/* Checks for and prints error encountered */
#define CUDA_CHECK(call)                           \
    do                                             \
    {                                              \
        cudaError_t error = (call);                \
        if (error != cudaSuccess)                  \
        {                                          \
            std::cerr << "CUDA Error: "            \
                      << cudaGetErrorString(error) \
                      << " at " << __FILE__        \
                      << ":" << __LINE__           \
                      << std::endl;                \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    } while (0)

constexpr int WARMUP_RUNS = 3;
constexpr int MEASUREMENTS = 10;

/*
 *  Call method from cpu to execute on gpu to add two vectors
 *  Each thread performs an add operation
 *  Explicitly limit operations to N threads to avoid access errors
 */
__global__ void vector_add(const float *A, const float *B, float *C, int N)
{
    // Global thread index = (block index) * (threads per block) + thread index within block
    int thread_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_i < N)
    {
        C[thread_i] = A[thread_i] + B[thread_i];
    }
}

int main()
{
    const int N = 10'000'000;
    const size_t bytes = N * sizeof(float);

    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N, 0.0f);

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    std::vector<int> block_sizes = {32, 64, 128, 256, 512};

    for (int threads_per_block : block_sizes)
    {
        const int num_blocks =
            (N + threads_per_block - 1) / threads_per_block;

        // Warmup runs
        for (int run = 0; run < WARMUP_RUNS; ++run)
        {
            vector_add<<<num_blocks, threads_per_block>>>(dA, dB, dC, N);

            // Check for any errors during launch
            CUDA_CHECK(cudaGetLastError());

            // Wait for all threads to complete vector_add
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<float> runtimes;
        runtimes.reserve(MEASUREMENTS);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Measured runs
        for (int run = 0; run < MEASUREMENTS; ++run)
        {
            CUDA_CHECK(cudaEventRecord(start));

            vector_add<<<num_blocks, threads_per_block>>>(dA, dB, dC, N);

            // Check for any errors during launch
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(stop));

            // Wait for all threads to complete vector_add
            CUDA_CHECK(cudaEventSynchronize(stop));

            float milliseconds = 0.0f;

            CUDA_CHECK(
                cudaEventElapsedTime(
                    &milliseconds,
                    start,
                    stop));

            runtimes.push_back(milliseconds);
        }

        std::sort(runtimes.begin(), runtimes.end());

        const float median =
            (runtimes[MEASUREMENTS / 2 - 1] +
             runtimes[MEASUREMENTS / 2]) /
            2.0f;

        std::cout << "Threads/block: " << threads_per_block
                  << ", Blocks: " << num_blocks
                  << ", Median Time: " << median
                  << " ms\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}