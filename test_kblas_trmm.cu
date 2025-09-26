#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

// Error checking macros
#define CHECK_CUDA(call)                                                                                                   \
    do                                                                                                                     \
    {                                                                                                                      \
        cudaError_t err = call;                                                                                            \
        if (err != cudaSuccess)                                                                                            \
        {                                                                                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1);                                                                                                       \
        }                                                                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                                                \
    do                                                                                                    \
    {                                                                                                     \
        cublasStatus_t stat = call;                                                                       \
        if (stat != CUBLAS_STATUS_SUCCESS)                                                                \
        {                                                                                                 \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << stat << std::endl; \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while (0)

// Constants and helper macros
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y

// Global variables
int kblas_trmm_ib_custom = 128;
int kblas_trmm_ib_cublas = 128;
int kblas_trmm_ib_data = 512;
bool kblas_trmm_use_custom = 0;

#define SIMPLE_SIZE(n) (((n) < WARP) || (((n) % WARP == 0) && ((n) <= kblas_trmm_ib_cublas)))
#define SIMPLE_SIZE_DATA(n) ((n) <= kblas_trmm_ib_data)

// Helper functions for different data types
template <typename T>
__device__ __host__ T make_zero();
template <>
__device__ __host__ float make_zero<float>() { return 0.0f; }
template <>
__device__ __host__ double make_zero<double>() { return 0.0; }

template <typename T>
__device__ __host__ T make_one();
template <>
__device__ __host__ float make_one<float>() { return 1.0f; }
template <>
__device__ __host__ double make_one<double>() { return 1.0; }

template <typename T>
__device__ T conjugate(T x) { return x; }

#define FMA(a, b, c) fmaf(a, b, c)

// Shuffle function wrapper
template <typename T>
__device__ T shfl(T var, int srcLane)
{
    return __shfl_sync(0xffffffff, var, srcLane);
}

// Custom TRMM kernel for left side multiplication
template <typename T, int WARPS_PER_BLOCK, int B_COLS_PER_WARP, bool LOWER, bool TRANS, bool CONJG>
__global__ void trmm_mul32_L(int M, int N, T alpha, const T *__restrict__ A, int incA, T *B, int incB, int mb)
{
    // if (tx == 0 && ty == 0 && blockIdx.x == 0)
    // {
    //     if (((uintptr_t)A) % 32 == 0)
    //         printf("A is 32-byte aligned\n");
    //     else
    //         printf("A is NOT 32-byte aligned (addr=%p)\n", A);

    //     if (((uintptr_t)B) % 32 == 0)
    //         printf("B is 32-byte aligned\n");
    //     else
    //         printf("B is NOT 32-byte aligned (addr=%p)\n", B);
    // }

    const int A_COL_PER_WARP = WARP / WARPS_PER_BLOCK;
    const bool forward = (LOWER == TRANS);

    int txyw = tx + ty * WARP1, txyiA = tx + ty * incA, txyiB = tx + ty * incB;

    // Setup shared memory
    __shared__ T sA[WARP * WARP1]; // Strided to avoid bank conflict
    T rB[B_COLS_PER_WARP], rBj[B_COLS_PER_WARP], s[B_COLS_PER_WARP], a[4], b[4], *sAA, *BB;
    int c, j, r, l, i, startB = 0, active_col;

    for (startB = 0; startB < N; startB += gridDim.x * WARPS_PER_BLOCK * B_COLS_PER_WARP)
    {
        if ((startB + blockIdx.x * WARPS_PER_BLOCK * B_COLS_PER_WARP) >= N)
            return;

        BB = B + (startB + blockIdx.x * WARPS_PER_BLOCK * B_COLS_PER_WARP) * incB;
        active_col = 0; // An inactive warp will still contribute to data fetching but not to computation

#pragma unroll
        for (l = 0; l < B_COLS_PER_WARP; l++)
            active_col += ((startB + blockIdx.x * (WARPS_PER_BLOCK * B_COLS_PER_WARP) + ty + l * WARPS_PER_BLOCK) < N);

        for (c = (forward ? 0 : mb - 1); (forward && (c < mb)) || (!forward && (c > -1)); c += (forward ? 1 : -1))
        {
#pragma unroll
            for (l = 0; l < B_COLS_PER_WARP; l++)
                s[l] = make_zero<T>();

// Load A(c,c) from global to shared mem
#pragma unroll
            for (l = 0; l < A_COL_PER_WARP; l++)
                sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * c * (incA + 1) + l * WARPS_PER_BLOCK * incA];

// Load B(c) into registers
#pragma unroll
            for (l = 0; l < B_COLS_PER_WARP; l++)
                if (active_col > l)
                    rB[l] = BB[txyiB + WARP * c + l * WARPS_PER_BLOCK * incB];

            __syncthreads();

            // Perform trmm on shared mem
            if (active_col > 0)
            {
                if (forward)
                {
#pragma unroll
                    for (j = 0; j < WARP; j++)
                    {
#pragma unroll
                        for (l = 0; l < B_COLS_PER_WARP; l++)
                            rBj[l] = shfl(rB[l], j);
                        if (j >= tx)
                        {
                            a[0] = CONJG ? conjugate(sA[j + tx * WARP1]) : sA[j + tx * WARP1];
#pragma unroll
                            for (l = 0; l < B_COLS_PER_WARP; l++)
                                s[l] = FMA(a[0], rBj[l], s[l]);
                        }
                    }
                }
                else
                {
#pragma unroll
                    for (j = WARP - 1; j > -1; j--)
                    {
#pragma unroll
                        for (l = 0; l < B_COLS_PER_WARP; l++)
                            rBj[l] = shfl(rB[l], j);
                        if (j <= tx)
                        {
                            a[0] = CONJG ? conjugate(sA[tx + j * WARP1]) : sA[tx + j * WARP1];
#pragma unroll
                            for (l = 0; l < B_COLS_PER_WARP; l++)
                                s[l] = FMA(a[0], rBj[l], s[l]);
                        }
                    }
                }
            }
            __syncthreads();

            for (r = (forward ? c + 1 : 0); (forward && (r < mb)) || (!forward && (r < c)); r++)
            {
#pragma unroll
                for (l = 0; l < A_COL_PER_WARP; l++)
                {
                    if (TRANS) // load A(r,c)
                        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
                    else // load A(c,r)
                        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (c + r * incA) + l * WARPS_PER_BLOCK * incA];
                }
// Load B(r)
#pragma unroll
                for (l = 0; l < B_COLS_PER_WARP; l++)
                    if (active_col > l)
                        rB[l] = BB[txyiB + WARP * r + l * WARPS_PER_BLOCK * incB];
                __syncthreads();

                // GEMM A(r,c)|A(c,r) & B(r) onto B(c) held at s
                if (active_col > 0)
                {
                    if (TRANS)
                        sAA = sA + tx * WARP1;
                    else
                        sAA = sA + tx;
#pragma unroll
                    for (j = 0; j < WARP; j += 4)
                    {
                        if (TRANS)
                        {
#pragma unroll
                            for (i = 0; i < 4; i++)
                                a[i] = CONJG ? conjugate(sAA[j + i]) : sAA[j + i];
                        }
                        else
                        {
#pragma unroll
                            for (i = 0; i < 4; i++)
                                a[i] = sAA[(j + i) * WARP1];
                        }

#pragma unroll
                        for (l = 0; l < B_COLS_PER_WARP; l++)
                        {
#pragma unroll
                            for (i = 0; i < 4; i++)
                                b[i] = shfl(rB[l], j + i);
#pragma unroll
                            for (i = 0; i < 4; i++)
                                s[l] = FMA(a[i], b[i], s[l]);
                        }
                    }
                }
                __syncthreads();
            }
// Store back B(c) to global mem
#pragma unroll
            for (l = 0; l < B_COLS_PER_WARP; l++)
            {
                if (active_col > l)
                {
                    BB[txyiB + WARP * c + l * WARPS_PER_BLOCK * incB] = alpha * s[l];
                }
            }
        }
    }
}

// Dummy right-side kernel (simplified for this example)
template <typename T, int WARPS_PER_BLOCK, int B_ROWS_PER_WARP, bool LOWER, bool TRANS, bool CONJG>
__global__ void trmm_mul32_R(int M, int N, T alpha, const T *__restrict__ A, int incA, T *B, int incB, int nb)
{
    // Simplified implementation - just call cuBLAS equivalent
    // For the full implementation, refer to the original code
}

// cuBLAS wrapper functions
cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                           float *B, int ldb)
{
    return cublasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, B, ldb);
}

cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                           double *B, int ldb)
{
    return cublasDtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, B, ldb);
}

// Custom TRMM implementation
template <class T>
cublasStatus_t Xtrmm(cublasHandle_t handle,
                     cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag,
                     int m, int n,
                     const T *alpha, const T *A, int incA,
                     T *B, int incB)
{
    typedef void (*trmm_kernels_type)(int M, int N, T alpha, const T *A, int incA, T *B, int incB, int mb);

#define WARPS_PER_BLOCK 8
#define B_COLS_PER_WARP 1

    trmm_kernels_type trmm_kernels[8] = {// T, WARPS_PER_BLOCK, B_COLS_PER_WARP, LEFT, LOWER, TRANS, CONJG
                                         trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, true, false, false>,
                                         trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, true, true, false>,
                                         trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false, false>,
                                         trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, true, false>,
                                         trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, true, false, false>,
                                         trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, true, true, false>,
                                         trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false, false>,
                                         trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, true, false>};

    cudaStream_t curStream;
    CHECK_CUBLAS(cublasGetStream(handle, &curStream));

    if (((side == CUBLAS_SIDE_LEFT) && (m % WARP == 0)) || ((side == CUBLAS_SIDE_RIGHT) && (n % WARP == 0)))
    {
        int func_idx = 4 * (side == CUBLAS_SIDE_RIGHT) + 2 * (uplo == CUBLAS_FILL_MODE_UPPER) + (trans != CUBLAS_OP_N);
        dim3 blockDim(WARP, WARPS_PER_BLOCK);
        printf("func_idx: %d", func_idx);
        dim3 gridDim(
            (side == CUBLAS_SIDE_LEFT) * (n / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (n % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0)) +
                (side == CUBLAS_SIDE_RIGHT) * (m / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (m % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0)),
            1);
        int mb = (side == CUBLAS_SIDE_LEFT) * m / WARP + (side == CUBLAS_SIDE_RIGHT) * n / WARP;



        trmm_kernels[func_idx]<<<gridDim, blockDim, 0, curStream>>>(m, n, *alpha, A, incA, B, incB, mb);
        CHECK_CUDA(cudaGetLastError());
    }
    else
    {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Error comparison function (equivalent to torch.allclose)
template <typename T>
bool compareResults(const T *a, const T *b, int size, T rtol = 1e-5, T atol = 1e-8)
{
    T max_error = 0.0;
    for (int i = 0; i < size; i++)
    {

        T diff = std::abs(a[i] - b[i]);
        if (i < 100)
            printf("diff:%.9f %.9f %.9f\n", a[i], b[i], diff);
        T tolerance = atol + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        if (diff > tolerance)
        {
            return false;
        }
        max_error = std::max(max_error, diff);
    }
    std::cout << "Maximum error between results: " << max_error << std::endl;
    return true;
}

// Initialize matrix with random values
template <typename T>
void initializeMatrix(T *matrix, int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1.0, 1.0);

    for (int i = 0; i < size; i++)
    {
        matrix[i] = dis(gen);
    }
}

// Make matrix triangular
template <typename T>
void makeTriangular(T *matrix, int n, bool lower)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (lower && j > i)
            {
                matrix[i * n + j] = 0.0;
            }
            else if (!lower && j < i)
            {
                matrix[i * n + j] = 0.0;
            }
        }
        // Make diagonal non-zero
        matrix[i * n + i] = std::abs(matrix[i * n + i]) + 1.0;
    }

    int rows = std::min(n, 10);
    int cols = std::min(n, 10);

    std::cout << "\nTop-left 10x10 block of matrix:\n";
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << std::setw(8) << std::setprecision(4) << matrix[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main()
{
    const int N = 4096*5;
    const int M = 4096*5;
    const int warmup_runs = 5;
    const int test_runs = 10;

    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate host memory
    std::vector<float> h_A(N * N);
    std::vector<float> h_B_cublas(M * N);
    std::vector<float> h_B_custom(M * N);
    std::vector<float> h_B_original(M * N);

    if (((uintptr_t)h_A.data()) % 32 == 0)
    {
        std::cout << "h_A is 32-byte aligned on host" << std::endl;
    }
    else
    {
        std::cout << "h_A is NOT 32-byte aligned on host" << std::endl;
    }

    // Initialize matrices
    initializeMatrix(h_A.data(), N * N);
    initializeMatrix(h_B_original.data(), M * N);
    makeTriangular(h_A.data(), N, true); // Make A lower triangular

    // Copy B for both tests
    std::copy(h_B_original.begin(), h_B_original.end(), h_B_cublas.begin());
    std::copy(h_B_original.begin(), h_B_original.end(), h_B_custom.begin());

    // Allocate device memory
    float *d_A, *d_B_cublas, *d_B_custom;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_cublas, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_custom, M * N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Parameters for TRMM
    const float alpha = 1.0f;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Warming up cuBLAS..." << std::endl;
    // Warmup cuBLAS
    for (int i = 0; i < warmup_runs; i++)
    {
        CHECK_CUDA(cudaMemcpy(d_B_cublas, h_B_original.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUBLAS(cublasStrmm(handle, side, uplo, trans, diag, M, N, &alpha, d_A, N, d_B_cublas, M, d_B_cublas, M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Benchmarking cuBLAS TRMM..." << std::endl;
    // Benchmark cuBLAS
    float total_time_cublas = 0.0f;
    for (int i = 0; i < test_runs; i++)
    {
        CHECK_CUDA(cudaMemcpy(d_B_cublas, h_B_original.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasStrmm(handle, side, uplo, trans, diag, M, N, &alpha, d_A, N, d_B_cublas, M, d_B_cublas, M));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time_cublas += milliseconds;
    }

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_B_cublas.data(), d_B_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Warming up custom kernel..." << std::endl;
    // Warmup custom kernel
    kblas_trmm_use_custom = true;
    for (int i = 0; i < warmup_runs; i++)
    {
        CHECK_CUDA(cudaMemcpy(d_B_custom, h_B_original.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
        Xtrmm(handle, side, uplo, trans, diag, M, N, &alpha, d_A, N, d_B_custom, M);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Benchmarking custom TRMM kernel..." << std::endl;
    // Benchmark custom kernel
    float total_time_custom = 0.0f;
    for (int i = 0; i < test_runs; i++)
    {
        CHECK_CUDA(cudaMemcpy(d_B_custom, h_B_original.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(start));
        Xtrmm(handle, side, uplo, trans, diag, M, N, &alpha, d_A, N, d_B_custom, M);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time_custom += milliseconds;
    }

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_B_custom.data(), d_B_custom, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate averages
    float avg_time_cublas = total_time_cublas / test_runs;
    float avg_time_custom = total_time_custom / test_runs;
    float speedup = avg_time_cublas / avg_time_custom;

    // Print results
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << std::endl;
    std::cout << "\nTiming Results (average of " << test_runs << " runs):" << std::endl;
    std::cout << "cuBLAS TRMM:      " << std::fixed << std::setprecision(6) << avg_time_cublas << " ms" << std::endl;
    std::cout << "Custom kernel:    " << std::fixed << std::setprecision(6) << avg_time_custom << " ms" << std::endl;

    if (speedup > 1.0f)
    {
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(5) << speedup << "x (Custom kernel is faster)" << std::endl;
    }
    else
    {
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(5) << speedup << "x (cuBLAS is faster)" << std::endl;
    }

    // Verify results
    std::cout << "\nVerification:" << std::endl;
    bool results_match = compareResults(h_B_cublas.data(), h_B_custom.data(), M * N);
    if (results_match)
    {
        std::cout << "✓ Results match (within numerical precision)" << std::endl;
    }
    else
    {
        std::cout << "✗ Results do not match!" << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B_cublas));
    CHECK_CUDA(cudaFree(d_B_custom));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}