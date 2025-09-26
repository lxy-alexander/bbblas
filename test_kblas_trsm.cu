#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#define WARP 32
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
        exit(1); \
    } \
} while(0)

template<typename T>
__device__ __forceinline__ T make_zero() {
    return T(0.0);
}

template<typename T>
__device__ __forceinline__ T conjugate(T x) {
    return x; // For double, conjugate is identity
}

template<typename T>
__device__ __forceinline__ T FMA(T a, T b, T c) {
    return fma(a, b, c);
}

template<typename T>
__device__ __forceinline__ T shfl(T var, int srcLane) {
    return __shfl_sync(0xffffffff, var, srcLane);
}

// Custom TRSM kernel based on the provided template
template<typename T, int WARPS_PER_BLOCK, bool LOWER, bool TRANS, bool CONJG, bool UNIT>
__global__ void trsm_mul32_L(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb)
{
    const int A_COLS_PER_WARP = WARP / WARPS_PER_BLOCK;
    const bool forward = (LOWER != TRANS);
    const short WARP1 = (TRANS ? 33 : 32);

    //setup shared memory
    __shared__ T sA[WARP * 33]; // Use 33 to avoid bank conflicts

    int tx = threadIdx.x % WARP;
    int ty = threadIdx.x / WARP;
    int txyw = tx + ty * WARP1, txyiA = tx + ty * incA, txyiB = tx + ty * incB, jtxw;
    int l, c, r, startB = 0, i;
    T rB, s, rBj, a[4], b[4], *sAA, *BB;

    for(startB = 0; startB < N; startB += gridDim.x * WARPS_PER_BLOCK)
    {
        if( (blockIdx.x * WARPS_PER_BLOCK + startB) >= N)
            return;

        BB = B + (blockIdx.x * WARPS_PER_BLOCK + startB) * incB;

        //checking boundary case, the column indices of B this warp is computing
        //if not active, this warp will only participate in fetching A sub-matrices, will not compute
        bool active = ( (blockIdx.x * WARPS_PER_BLOCK + startB + ty) < N );

        for(c = (forward ? 0 : mb-1); (forward && c < mb) || (!forward && c >= 0); c += (forward ? 1 : -1))
        {
            s = make_zero<T>();

            for(r = (forward ? 0 : mb-1); (forward && r < c) || (!forward && r > c); r += (forward ? 1 : -1))
            {
                #pragma unroll
                for(l = 0; l < A_COLS_PER_WARP; l++){
                    if(TRANS)
                        //load A(r,c)
                        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
                    else
                        //load A(c,r)
                        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = __ldg(&(A[txyiA + WARP * (c + r * incA) + l * WARPS_PER_BLOCK * incA]));
                }
                //load B(r)
                if(active)
                    rB = __ldg(&(BB[txyiB + WARP * r]));

                __syncthreads();
                if(active){
                    //gemm A(r,c)/A(c,r) & B(r) onto B(c) held at s
                    if(TRANS)
                        sAA = sA + tx*WARP1;
                    else
                        sAA = sA + tx;
                    #pragma unroll
                    for(int j = 0; j < WARP; j+=4){
                        if(TRANS){
                            #pragma unroll
                            for(i = 0; i < 4; i++)
                                a[i] = CONJG ? conjugate(sAA[j + i]) : sAA[j + i];
                        }else{
                            #pragma unroll
                            for(i = 0; i < 4; i++)
                                a[i] = sAA[(j + i)*WARP1];
                        }
                        #pragma unroll
                        for(i = 0; i < 4; i++)
                            b[i] = shfl(rB, j + i);
                        #pragma unroll
                        for(i = 0; i < 4; i++)
                            s = FMA( a[i], b[i], s );
                    }
                }
                __syncthreads();
            }

            //load A(c,c) from global to shared mem
            #pragma unroll
            for(l = 0; l < A_COLS_PER_WARP; l++){
                sA[txyw + l * WARPS_PER_BLOCK * WARP1] = __ldg(&(A[txyiA + WARP * c * (incA + 1) + l * WARPS_PER_BLOCK * incA]));
            }

            //load B(c) into registers
            if(active){
                rB = __ldg(&(BB[txyiB + WARP * c]));
            }
            __syncthreads();
            if(active)
            {
                //perform trsm on shared mem
                if(!LOWER && TRANS)
                    jtxw = tx * WARP1;
                else
                if(!LOWER && !TRANS)
                    jtxw = tx         + (WARP - 1) * WARP1;
                else
                if(LOWER && TRANS)
                    jtxw = tx * WARP1 + (WARP - 1);
                else
                if(LOWER && !TRANS)
                    jtxw = tx;

                #pragma unroll
                for(int j = (forward ? 0 : WARP-1); (forward && (j < WARP)) || (!forward && (j >= 0)); j += (forward ? 1 : -1)){
                    if(j == tx){
                        rB = FMA(alpha, rB, -s);
                        if(!UNIT){
                            a[0] = (TRANS && CONJG) ? conjugate(sA[tx * (WARP1+1)]) : sA[tx * (WARP1+1)];//diagonal element
                            rB = rB / a[0];
                        }
                    }
                    rBj = shfl(rB, j);

                    if( (forward && (j < tx)) || (!forward && (j > tx)) ){
                        a[0] = (TRANS && CONJG) ? conjugate(sA[jtxw]) : sA[jtxw];
                        s = FMA(a[0], rBj, s);
                    }
                    jtxw += (TRANS ? 1 : WARP1) * (forward ? 1 : -1);
                }

                //store back B(c) to global mem
                BB[txyiB + WARP * c] = rB;
            }
            __syncthreads();
        }
    }
}

// Error checking function equivalent to torch.allclose
bool compareResults(const double* a, const double* b, int size, double rtol = 1e-5, double atol = 1e-8) {
    double max_error = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = std::abs(a[i] - b[i]);
        double tolerance = atol + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        if (diff > tolerance) {
            return false;
        }
        max_error = std::max(max_error, diff);
    }
    std::cout << "Maximum error between results: " << std::scientific << max_error << std::endl;
    return true;
}

void initializeMatrix(double* matrix, int rows, int cols, bool lower_triangular = false) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (lower_triangular && j > i) {
                matrix[i * cols + j] = 0.0;
            } else {
                matrix[i * cols + j] = dist(gen);
                if (i == j && lower_triangular) {
                    matrix[i * cols + j] = std::abs(matrix[i * cols + j]) + 1.0; // Ensure diagonal dominance
                }
            }
        }
    }
}

void initializeVector(double* vector, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < size; i++) {
        vector[i] = dist(gen);
    }
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const double alpha = 1.0;
    const int warmup_runs = 5;
    const int benchmark_runs = 20;
    
    // Host memory allocation
    std::vector<double> h_A(M * M);
    std::vector<double> h_B_cublas(M * N);
    std::vector<double> h_B_custom(M * N);
    std::vector<double> h_B_original(M * N);
    
    // Initialize matrices
    initializeMatrix(h_A.data(), M, M, true); // Lower triangular matrix
    initializeVector(h_B_original.data(), M * N);
    
    // Copy original B for both tests
    std::copy(h_B_original.begin(), h_B_original.end(), h_B_cublas.begin());
    std::copy(h_B_original.begin(), h_B_original.end(), h_B_custom.begin());
    
    // Device memory allocation
    double *d_A, *d_B_cublas, *d_B_custom;
    CHECK_CUDA(cudaMalloc(&d_A, M * M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B_cublas, M * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_B_custom, M * N * sizeof(double)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * M * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_cublas, h_B_cublas.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_custom, h_B_custom.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // cuBLAS setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up cuBLAS
    std::cout << "Warming up cuBLAS..." << std::endl;
    for (int i = 0; i < warmup_runs; i++) {
        CHECK_CUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                M, N, &alpha, d_A, M, d_B_cublas, M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark cuBLAS
    std::cout << "Benchmarking cuBLAS dtrsm..." << std::endl;
    float cublas_total_time = 0.0f;
    
    for (int i = 0; i < benchmark_runs; i++) {
        // Reset B matrix
        CHECK_CUDA(cudaMemcpy(d_B_cublas, h_B_original.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                M, N, &alpha, d_A, M, d_B_cublas, M));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        cublas_total_time += milliseconds;
    }
    float cublas_avg_time = cublas_total_time / benchmark_runs;
    
    // Copy cuBLAS result back to host for verification
    CHECK_CUDA(cudaMemcpy(h_B_cublas.data(), d_B_cublas, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Custom kernel setup
    const int WARPS_PER_BLOCK = 4;
    const int THREADS_PER_BLOCK = WARP * WARPS_PER_BLOCK;
    const int mb = (M + WARP - 1) / WARP;
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    
    // Warm-up custom kernel
    std::cout << "Warming up custom kernel..." << std::endl;
    for (int i = 0; i < warmup_runs; i++) {
        trsm_mul32_L<double, WARPS_PER_BLOCK, true, false, false, false>
            <<<gridDim, blockDim>>>(M, N, alpha, d_A, M, d_B_custom, M, mb);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark custom kernel
    std::cout << "Benchmarking custom trsm kernel..." << std::endl;
    float custom_total_time = 0.0f;
    
    for (int i = 0; i < benchmark_runs; i++) {
        // Reset B matrix
        CHECK_CUDA(cudaMemcpy(d_B_custom, h_B_original.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaEventRecord(start));
        trsm_mul32_L<double, WARPS_PER_BLOCK, true, false, false, false>
            <<<gridDim, blockDim>>>(M, N, alpha, d_A, M, d_B_custom, M, mb);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        custom_total_time += milliseconds;
    }
    float custom_avg_time = custom_total_time / benchmark_runs;
    
    // Copy custom kernel result back to host for verification
    CHECK_CUDA(cudaMemcpy(h_B_custom.data(), d_B_custom, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Performance results
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << M << std::endl;
    std::cout << "Vector size: " << N << std::endl;
    std::cout << std::endl;
    
    std::cout << "Timing Results (average of " << benchmark_runs << " runs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "cuBLAS dtrsm:     " << cublas_avg_time << " ms" << std::endl;
    std::cout << "Custom kernel:    " << custom_avg_time << " ms" << std::endl;
    std::cout << std::endl;
    
    // Calculate speedup
    float speedup = cublas_avg_time / custom_avg_time;
    std::cout << "Speedup: " << std::setprecision(5) << speedup << "x ";
    if (speedup > 1.0f) {
        std::cout << "(Custom kernel is faster)" << std::endl;
    } else {
        std::cout << "(cuBLAS is faster)" << std::endl;
    }
    std::cout << std::endl;
    
    // Verification
    std::cout << "Verification:" << std::endl;
    bool results_match = compareResults(h_B_cublas.data(), h_B_custom.data(), M * N);
    if (results_match) {
        std::cout << "✓ Results match (within numerical precision)" << std::endl;
    } else {
        std::cout << "✗ Results do not match!" << std::endl;
    }
    
    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B_cublas));
    CHECK_CUDA(cudaFree(d_B_custom));
    
    return 0;
}