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
 *  Call method from cpu to execute on gpu to copy a row major matrix using coalesced access
 */
__global__ void coalesced_copy(const float *A, float *B, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        int coalesced_index = N * row + col;
        B[coalesced_index] = A[coalesced_index];
    }
}

/*
 *  Call method from cpu to execute on gpu to copy a row major matrix using strided access
 */
__global__ void strided_copy(const float *A, float *B, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        int strided_index = N * col + row;
        B[strided_index] = A[strided_index];
    }
}

int main()
{
    const int N = 8192;
    const size_t num_elements = static_cast<size_t>(N) * N;
    const size_t bytes = num_elements * sizeof(float);

    std::vector<float> A(num_elements, 1.0f);
    std::vector<float> B(num_elements, 0.0f);

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

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    /*
     *  Warmup runs for both kernels
     */
    for (int run = 0; run < WARMUP_RUNS; run++)
    {
        coalesced_copy<<<grid, block>>>(dA, dB, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        strided_copy<<<grid, block>>>(dA, dB, N);
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

    /*
     *  Measure coalesced and strided kernel runtimes
     */
    for (int run = 0; run < MEASUREMENTS; run++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        coalesced_copy<<<grid, block>>>(dA, dB, N);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float coalesced_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &coalesced_ms,
            start,
            stop
        ));

        coalesced_runtimes.push_back(coalesced_ms);

        CUDA_CHECK(cudaEventRecord(start));

        strided_copy<<<grid, block>>>(dA, dB, N);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float strided_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &strided_ms,
            start,
            stop
        ));

        strided_runtimes.push_back(strided_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(
        coalesced_runtimes.begin(),
        coalesced_runtimes.end()
    );

    std::sort(
        strided_runtimes.begin(),
        strided_runtimes.end()
    );

    const float coalesced_median =
        (coalesced_runtimes[MEASUREMENTS / 2 - 1] +
         coalesced_runtimes[MEASUREMENTS / 2]) /
        2.0f;

    const float strided_median =
        (strided_runtimes[MEASUREMENTS / 2 - 1] +
         strided_runtimes[MEASUREMENTS / 2]) /
        2.0f;

    std::cout << "Coalesced Median Time: "
              << coalesced_median
              << " ms\n";

    std::cout << "Strided Median Time: "
              << strided_median
              << " ms\n";

    std::cout << "Slowdown: "
              << strided_median / coalesced_median
              << "x\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));

    return 0;
}