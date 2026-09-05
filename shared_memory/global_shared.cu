#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// Checks for and prints error encountered
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
constexpr int THREADS_PER_BLOCK = 256;

/*
 * Call method from cpu to execute on gpu where each thread repeatedly
 * reads values directly from global memory
 */
__global__ void global_reuse(
    const float *A,
    float *B,
    int N,
    int reuse)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        int block_start = blockIdx.x * blockDim.x;

        float sum = 0.0f;

        // Each thread reads reuse number of values directly from global memory
        for (int offset = 0; offset < reuse; offset++)
        {
            int neighbor =
                block_start +
                (threadIdx.x + offset) % blockDim.x;

            sum += A[neighbor];
        }

        B[i] = sum;
    }
}

/*
 * Call method from cpu to execute on gpu where each thread first stores its value
 * in shared memory and then repeatedly reads values from shared memory
 */
__global__ void shared_reuse(
    const float *A,
    float *B,
    int N,
    int reuse)
{
    __shared__ float shared_data[THREADS_PER_BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one value from global memory into shared memory
    shared_data[threadIdx.x] = A[i];

    // Wait until every thread in the block has finished loading shared memory
    __syncthreads();

    float sum = 0.0f;

    // Each thread reuses values already stored in shared memory
    for (int offset = 0; offset < reuse; offset++)
    {
        int neighbor =
            (threadIdx.x + offset) % blockDim.x;

        sum += shared_data[neighbor];
    }

    B[i] = sum;
}

/*
 * Calculate median runtime
 */
float median(std::vector<float> runtimes)
{
    std::sort(
        runtimes.begin(),
        runtimes.end()
    );

    return (
        runtimes[MEASUREMENTS / 2 - 1] +
        runtimes[MEASUREMENTS / 2]
    ) / 2.0f;
}

/*
 * Measure one execution of the global memory kernel
 */
float measure_global(
    const float *dA,
    float *dB,
    int N,
    int blocks,
    int reuse,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    CUDA_CHECK(cudaEventRecord(start));

    global_reuse<<<blocks, THREADS_PER_BLOCK>>>(
        dA,
        dB,
        N,
        reuse
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    ));

    return milliseconds;
}

/*
 * Measure one execution of the shared memory kernel
 */
float measure_shared(
    const float *dA,
    float *dB,
    int N,
    int blocks,
    int reuse,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    CUDA_CHECK(cudaEventRecord(start));

    shared_reuse<<<blocks, THREADS_PER_BLOCK>>>(
        dA,
        dB,
        N,
        reuse
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    ));

    return milliseconds;
}

int main()
{
    // Keep N divisible by THREADS_PER_BLOCK for this experiment
    const int N = 1 << 24;

    const size_t bytes =
        static_cast<size_t>(N) * sizeof(float);

    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 0.0f);

    float *dA = nullptr;
    float *dB = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));

    CUDA_CHECK(cudaMemcpy(
        dA,
        A.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    const int blocks =
        (N + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    // Different amounts of data reuse to test
    const std::vector<int> reuse_sizes =
        {1, 2, 4, 8, 16, 32};

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout
        << "Reuse\tGlobal (ms)\tShared (ms)\tGlobal / Shared\n";

    // Test global and shared memory for different amounts of reuse
    for (int reuse : reuse_sizes)
    {
        // Warmup runs for both kernels
        for (int run = 0; run < WARMUP_RUNS; run++)
        {
            global_reuse<<<blocks, THREADS_PER_BLOCK>>>(
                dA,
                dB,
                N,
                reuse
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            shared_reuse<<<blocks, THREADS_PER_BLOCK>>>(
                dA,
                dB,
                N,
                reuse
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<float> global_runtimes;
        global_runtimes.reserve(MEASUREMENTS);

        std::vector<float> shared_runtimes;
        shared_runtimes.reserve(MEASUREMENTS);

        // Alternate measurement order to reduce ordering bias
        for (int run = 0; run < MEASUREMENTS; run++)
        {
            float global_ms = 0.0f;
            float shared_ms = 0.0f;

            if (run % 2 == 0)
            {
                // Even runs measure global first
                global_ms = measure_global(
                    dA,
                    dB,
                    N,
                    blocks,
                    reuse,
                    start,
                    stop
                );

                shared_ms = measure_shared(
                    dA,
                    dB,
                    N,
                    blocks,
                    reuse,
                    start,
                    stop
                );
            }
            else
            {
                // Odd runs measure shared first
                shared_ms = measure_shared(
                    dA,
                    dB,
                    N,
                    blocks,
                    reuse,
                    start,
                    stop
                );

                global_ms = measure_global(
                    dA,
                    dB,
                    N,
                    blocks,
                    reuse,
                    start,
                    stop
                );
            }

            global_runtimes.push_back(global_ms);
            shared_runtimes.push_back(shared_ms);
        }

        const float global_median =
            median(global_runtimes);

        const float shared_median =
            median(shared_runtimes);

        std::cout
            << reuse << "\t"
            << global_median << "\t\t"
            << shared_median << "\t\t"
            << global_median / shared_median
            << "x\n";
    }

    // Copy result back to cpu to verify correctness
    CUDA_CHECK(cudaMemcpy(
        B.data(),
        dB,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // Last experiment uses reuse = 32 and A contains only 1.0
    // so every output should contain 32.0
    bool passed = true;

    for (int i = 0; i < N; i++)
    {
        if (B[i] != 32.0f)
        {
            passed = false;
            break;
        }
    }

    std::cout
        << "\nCorrectness: "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));

    return 0;
}