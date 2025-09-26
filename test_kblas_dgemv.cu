#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <random>

// Helper function to create zero value for different types
template<typename T>
__device__ __host__ T make_zero() {
    return T(0.0);
}

// Custom atomicAdd for double (for older CUDA versions)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// For compute capability 6.0 and above, native atomicAdd for double is available
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template <class T, int nb, int tcol, int ept, int width, int ept_>
__global__ void
gemvn(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy, int mod_r, int mod_c, int threshold)
{
    const int   tx   = threadIdx.x ;
    const int   ty   = threadIdx.y ;
    const int   blkc = blockIdx.x ;
    const int   by  =   blockIdx.y;

    T res_1_    = make_zero<T>();
    T areg[ept];
    T breg[ept];

    __shared__ T la[nb * tcol];

    if(blkc == gridDim.x-1)
    {
        if(mod_r > 0){if(tx >= mod_r)return;}
    }

    int count = (cols/width)/gridDim.y + (by < (cols/width)%gridDim.y);

    {
        int start = by * ((cols/width)/gridDim.y) + min(by, (cols/width)%gridDim.y);

        A += nb * blkc;
        A += start * width * lda;
        x += start * width * incx;
        y += (blkc * nb) * incy;
    }

    if(by != gridDim.y-1){if(count == 0) return;}
    else {if(count == 0 && mod_c == 0) return;}

    const int j = ty * ept * lda + tx;

    if(count >= 2)
    {
        #pragma unroll
        for(int k = 0; k < ept; k++)
            areg[k] = A[j + k * lda];
        A += width * lda;
    }

    int Vblocks = 0;
    #pragma unroll
    for(Vblocks = 0; Vblocks < (count/2)*2; Vblocks+=2)
    {
        #pragma unroll
        for(int k = 0; k < ept; k++)
            breg[k] = A[j + k * lda];
        A += width * lda;

        #pragma unroll
        for(int k = 0; k < ept; k++)
            res_1_ += areg[k] * x[(ty * ept + k) * incx];
        x += width * incx;

        if(Vblocks != ((count/2)*2-2) )
        {
            #pragma unroll
            for(int k = 0; k < ept; k++)
                areg[k] = A[j + k * lda];
            A += width * lda;
        }

        #pragma unroll
        for(int k = 0; k < ept; k++)
            res_1_ += breg[k] * x[(ty * ept + k) * incx];
        x += width * incx;
    }

    if(count%2 >= 1)
    {
        #pragma unroll
        for(int k = 0; k < ept; k++)
            areg[k] = A[j + k * lda];
        A += width * lda;

        // process remaining block
        #pragma unroll
        for(int k = 0; k < ept; k++)
            res_1_ += areg[k] * x[(ty * ept + k) * incx];
        x += width * incx;
    }

    if(by == gridDim.y-1)
    {
        #pragma unroll
        for(int k = 0; k < ept; k++){breg[k] = make_zero<T>();}

        if(mod_c != 0)
        {
            if(ty < threshold)
            {
                #pragma unroll
                for(int k = 0; k < ept; k++)
                    breg[k] = A[j + k * lda];
            }
            else if(ty == threshold)
            {
                #pragma unroll
                for(int k = 0; k < ept_; k++)
                    breg[k] = A[j + k * lda];
            }

            // compute
            if(ty < threshold)
            {
                #pragma unroll
                for(int k = 0; k < ept; k++)
                    res_1_ += breg[k] * x[(ty * ept + k) * incx];
            }
            else if (ty == threshold)
            {
                #pragma unroll
                for(int k = 0; k < ept_; k++)
                    res_1_ += breg[k] * x[(ty * ept + k) * incx];
            }
        }
    }

    la[ty * nb + tx] = res_1_;
    __syncthreads();

    if(ty == 0)
    {
        res_1_ = make_zero<T>();
        #pragma unroll
        for(int k = 0; k < tcol; k++)
            res_1_ += la[k * nb + tx];
        // use atomics
        atomicAdd(&y[tx * incy], (alpha*res_1_));
        //y[tx] = alpha * res_1_ + res;
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << " - Error code: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    const int M = 4096;
    const int N = 4096;
    const double alpha = 1.0;
    const double beta = 0.0;
    const int incx = 1;
    const int incy = 1;
    const int lda = M;

    // Allocate host memory
    std::vector<double> h_A(M * N);
    std::vector<double> h_x(N);
    std::vector<double> h_y_cublas(M);
    std::vector<double> h_y_custom(M);

    // Initialize data with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < M * N; ++i) {
        h_A[i] = dis(gen);
    }
    for (int i = 0; i < N; ++i) {
        h_x[i] = dis(gen);
    }
    std::fill(h_y_cublas.begin(), h_y_cublas.end(), 0.0);
    std::fill(h_y_custom.begin(), h_y_custom.end(), 0.0);

    // Allocate device memory
    double *d_A, *d_x, *d_y_cublas, *d_y_custom;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_cublas, M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y_custom, M * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up for cuBLAS
    std::cout << "Warming up cuBLAS..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaMemcpy(d_y_cublas, h_y_cublas.data(), M * sizeof(double), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, lda, d_x, incx, &beta, d_y_cublas, incy));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark cuBLAS
    std::cout << "Benchmarking cuBLAS DGEMV..." << std::endl;
    float cublas_total_time = 0.0f;
    
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemcpy(d_y_cublas, h_y_cublas.data(), M * sizeof(double), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, lda, d_x, incx, &beta, d_y_cublas, incy));
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        cublas_total_time += elapsed_time;
    }
    
    float cublas_avg_time = cublas_total_time / 20.0f;

    // Custom kernel parameters (adjusted for 4096x4096)
    const int nb = 64;      // block size for rows
    const int tcol = 8;     // threads per column
    const int ept = 8;      // elements per thread
    const int width = tcol * ept;  // 64
    const int ept_ = ept;

    // Grid and block dimensions
    dim3 blockDim(nb, tcol);
    dim3 gridDim((M + nb - 1) / nb, (N + width - 1) / width);

    // Calculate mod values
    int mod_r = M % nb;
    int mod_c = N % width;
    int threshold = mod_c / ept;

    // Warm-up for custom kernel
    std::cout << "Warming up custom kernel..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaMemcpy(d_y_custom, h_y_custom.data(), M * sizeof(double), cudaMemcpyHostToDevice));
        gemvn<double, nb, tcol, ept, width, ept_><<<gridDim, blockDim>>>(
            M, N, alpha, d_A, lda, d_x, incx, beta, d_y_custom, incy, mod_r, mod_c, threshold);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark custom kernel
    std::cout << "Benchmarking custom GEMV kernel..." << std::endl;
    float custom_total_time = 0.0f;
    
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemcpy(d_y_custom, h_y_custom.data(), M * sizeof(double), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        gemvn<double, nb, tcol, ept, width, ept_><<<gridDim, blockDim>>>(
            M, N, alpha, d_A, lda, d_x, incx, beta, d_y_custom, incy, mod_r, mod_c, threshold);
        CUDA_CHECK(cudaEventRecord(stop));
        
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        custom_total_time += elapsed_time;
    }
    
    float custom_avg_time = custom_total_time / 20.0f;

    // Copy results back to host for verification
    CUDA_CHECK(cudaMemcpy(h_y_cublas.data(), d_y_cublas, M * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y_custom.data(), d_y_custom, M * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify results
    double max_error = 0.0;
    for (int i = 0; i < M; ++i) {
        double error = std::abs(h_y_cublas[i] - h_y_custom[i]);
        max_error = std::max(max_error, error);
    }

    // Print results
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << std::endl;
    std::cout << "Vector size: " << N << std::endl;
    std::cout << "\nTiming Results (average of 20 runs):" << std::endl;
    std::cout << "cuBLAS DGEMV:    " << cublas_avg_time << " ms" << std::endl;
    std::cout << "Custom kernel:   " << custom_avg_time << " ms" << std::endl;
    
    if (custom_avg_time > 0) {
        float speedup = cublas_avg_time / custom_avg_time;
        std::cout << "\nSpeedup: " << speedup << "x ";
        if (speedup > 1.0) {
            std::cout << "(Custom kernel is faster)";
        } else {
            std::cout << "(cuBLAS is faster)";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nVerification:" << std::endl;
    std::cout << "Maximum error between results: " << max_error << std::endl;
    if (max_error < 1e-10) {
        std::cout << "✓ Results match (within numerical precision)" << std::endl;
    } else if (max_error < 1e-6) {
        std::cout << "⚠ Results are close but show some numerical differences" << std::endl;
    } else {
        std::cout << "✗ Results differ significantly - possible implementation error" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y_cublas));
    CUDA_CHECK(cudaFree(d_y_custom));

    return 0;
}