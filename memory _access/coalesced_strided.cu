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
                      << cudaGetErrorString(error)  \
                      << " at " << __FILE__         \
                      << ":" << __LINE__            \
                      << std::endl;                 \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    } while (0)

constexpr int WARMUP_RUNS = 3;
constexpr int MEASUREMENTS = 10;

/*
 * Call method from cpu to execute on gpu to copy a row major matrix
 * using coalesced access
 */
__global__ void coalesced_copy(const float *A, float *B, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        int coalesced_index = N * row + col;

        B[coalesced_index] =
            A[coalesced_index];
    }
}

/*
 * Call method from cpu to execute on gpu to copy a row major matrix
 * using strided access
 */
__global__ void strided_copy(const float *A, float *B, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        int strided_index = N * col + row;

        B[strided_index] =
            A[strided_index];
    }
}

/*
 * Measure one execution of the coalesced kernel
 */
float measure_coalesced(
    const float *dA,
    float *dB,
    int N,
    dim3 grid,
    dim3 block,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    CUDA_CHECK(cudaEventRecord(start));

    coalesced_copy<<<grid, block>>>(
        dA,
        dB,
        N
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
 * Measure one execution of the strided kernel
 */
float measure_strided(
    const float *dA,
    float *dB,
    int N,
    dim3 grid,
    dim3 block,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    CUDA_CHECK(cudaEventRecord(start));

    strided_copy<<<grid, block>>>(
        dA,
        dB,
        N
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

int main()
{
    const int N = 8192;

    const size_t num_elements =
        static_cast<size_t>(N) * N;

    const size_t bytes =
        num_elements * sizeof(float);

    std::vector<float> A(
        num_elements,
        1.0f
    );

    std::vector<float> B(
        num_elements,
        0.0f
    );

    float *dA = nullptr;
    float *dB = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dA,
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        &dB,
        bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        dA,
        A.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 32 threads along x allows one warp to span one row of accesses
    dim3 block(32, 8);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    // Warmup runs for both kernels
    for (int run = 0; run < WARMUP_RUNS; run++)
    {
        coalesced_copy<<<grid, block>>>(
            dA,
            dB,
            N
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        strided_copy<<<grid, block>>>(
            dA,
            dB,
            N
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> coalesced_runtimes;
    coalesced_runtimes.reserve(MEASUREMENTS);

    std::vector<float> strided_runtimes;
    strided_runtimes.reserve(MEASUREMENTS);

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Alternate which kernel is measured first to reduce ordering bias
    for (int run = 0; run < MEASUREMENTS; run++)
    {
        float coalesced_ms = 0.0f;
        float strided_ms = 0.0f;

        if (run % 2 == 0)
        {
            // Even runs measure coalesced first
            coalesced_ms = measure_coalesced(
                dA,
                dB,
                N,
                grid,
                block,
                start,
                stop
            );

            strided_ms = measure_strided(
                dA,
                dB,
                N,
                grid,
                block,
                start,
                stop
            );
        }
        else
        {
            // Odd runs measure strided first
            strided_ms = measure_strided(
                dA,
                dB,
                N,
                grid,
                block,
                start,
                stop
            );

            coalesced_ms = measure_coalesced(
                dA,
                dB,
                N,
                grid,
                block,
                start,
                stop
            );
        }

        coalesced_runtimes.push_back(
            coalesced_ms
        );

        strided_runtimes.push_back(
            strided_ms
        );
    }

    const float coalesced_median =
        median(coalesced_runtimes);

    const float strided_median =
        median(strided_runtimes);

    std::cout
        << "Coalesced Median Time: "
        << coalesced_median
        << " ms\n";

    std::cout
        << "Strided Median Time: "
        << strided_median
        << " ms\n";

    std::cout
        << "Slowdown: "
        << strided_median / coalesced_median
        << "x\n";

    // Run coalesced kernel once and verify that the copy is correct
    coalesced_copy<<<grid, block>>>(
        dA,
        dB,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        B.data(),
        dB,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool passed = true;

    for (size_t i = 0; i < num_elements; i++)
    {
        if (B[i] != A[i])
        {
            passed = false;
            break;
        }
    }

    std::cout
        << "Correctness: "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));

    return 0;
}