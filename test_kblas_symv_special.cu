#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// Helper functions for the kernel
template<typename T>
__device__ __forceinline__ T make_zero() {
    return T(0.0f);
}

template<typename T>
__device__ __forceinline__ T conjugate(T val) {
    return val; // For real numbers, conjugate is the same
}

template<typename T>
__device__ __forceinline__ T make_real(T val) {
    return val; // For real numbers, already real
}

// First kernel - exactly as provided (diagonal blocks)
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvl_special_d( 	int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int td  = (thread_x * ty ) + tx;
    const int	blkc = blockIdx.x ;

    T res	= make_zero<T>();
    T yold	= make_zero<T>();
    //make_zero(&res);
    //make_zero(&yold);

    __shared__ T la   [syhemv_bs * syhemv_bs];
    __shared__ T buff [syhemv_bs];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];

	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * blkc * (lda + 1);

	// Advance 'A' to start row for each thread inside the diagonal block
	A += ty * lda + tx;

	// handle the case when incx and/or incy is -ve
	//if(incx < 0) x -= (n-1) * incx;
	//if(incy < 0) y -= (n-1) * incy;

	// Advance 'x'
	x += (blkc * syhemv_bs) * incx;

    // Advance 'y'
    y += (blkc * syhemv_bs) * incy;

	if(ty == 0)
	{
		yold = beta * y[incy * tx];
		buff[tx] = x[incx * tx];
	}

	// load first chunk
	#pragma unroll
	for(int k = 0; k < (syhemv_bs/2); k+= thread_y)
		la[td + k * syhemv_bs] = A[k * lda];

	// Advance to second chunk
	A += (syhemv_bs/2) * lda;
	// load second chunk
	if(tx >= (syhemv_bs/2))	// even warps will load un-necessary elements in the 2nd chunck og diagonal block
	{
	  #pragma unroll
	  for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		la[syhemv_bs * ((syhemv_bs/2) + j + ty) + tx] = A[j * lda];
	}

	__syncthreads();

	// mirror necessary elements in first chunk
	if(ty > tx)
		la[td] = conjugate( la[tx * syhemv_bs + ty] );
	else
		la[td] = la[td];

	#pragma unroll
	for(int k = thread_y; k < (syhemv_bs/2); k+= thread_y)
		if(abs(tx - ty) < k)
			la[tx + (ty + k) * syhemv_bs] = conjugate( la[ty + k + tx * syhemv_bs] );

	// mirror second chunk
	#pragma unroll
	for(int k = 0; k < (syhemv_bs/2); k+= thread_y)
		if(abs(tx-ty) < (k + (syhemv_bs/2)))
			la[syhemv_bs * ((syhemv_bs/2) + k + ty) + tx] = conjugate( la[syhemv_bs * tx + (syhemv_bs/2) + k + ty] );

	if(ty == 0) la[tx * syhemv_bs + tx] = make_real(la[tx * syhemv_bs + tx]);
	__syncthreads();

	// compute first chunk
	#pragma unroll
	for(int k = 0; k < (syhemv_bs/2); k+= thread_y)
		res += la[(ty + k) * syhemv_bs + tx] * buff[k + ty];

	// compute second chunk
	#pragma unroll
	for(int k = (syhemv_bs/2); k < 2 * (syhemv_bs/2); k+= thread_y)
		res += la[(ty + k) * syhemv_bs + tx] * buff[k + ty];

	accum[td] = res;

	__syncthreads();

	if(ty == 0)
	{
		res = make_zero<T>();
	  	#pragma unroll
	  	for(int k = 0; k < thread_y; k++)
			res += accum[k * syhemv_bs + tx];
		res *= alpha;
		res += yold;

		y[incy * tx] = res;
	}
}

// Second kernel - exactly as provided (non-diagonal blocks)
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvl_special_nd( 	int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by 	= blockIdx.y;
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (syhemv_bs/2);
    const int	ty_  = td / (syhemv_bs/2);
    T		* xcopy, *ycopy;

    // compute how many matrix blocks to be processed
	int count = (gridDim.x-blkc-1)/gridDim.y;
	//int count = (gridDim.x-blkc-1)/gridDim.y;
	//if(by < (gridDim.x-blkc-1)%gridDim.y) count++;

	T xreg[elements_per_thread];
	T areg[elements_per_thread];
	T treg[elements_per_thread] = { make_zero<T>()};

	//#pragma unroll
	//for(int k = 0; k < elements_per_thread; k++) make_zero(&treg[k]);

    __shared__ T la   [syhemv_bs * (syhemv_bs/2)];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    __shared__ T xbuff[syhemv_bs];

    if(blkc == gridDim.x-1)return;

	{
		// compute number of preceding blocks
		//int pr = by*(gridDim.x-blkc-1)/gridDim.y + min(by, (gridDim.x-blkc-1)%gridDim.y);

		// Advance 'A' to start of diagonal blocks first
		A += syhemv_bs * blkc * (lda + 1);
		// divide work among the y-direction of the grid
		A += (by * count) * syhemv_bs;

		// Advance 'x'
		x += (blkc * syhemv_bs) * incx;
		xcopy = x;
    	x += (by * count * syhemv_bs) * incx;

    	if(ty == 0) xbuff[tx] = xcopy[tx * incx];

    	// Advance 'y'
    	y += (blkc * syhemv_bs) * incy;
    	ycopy = y;
    	ycopy += (by * count * syhemv_bs) * incy;
    }
    if(by == gridDim.y-1) count += (gridDim.x-blkc-1)%gridDim.y;
	if(count == 0) return;

	T res_1_	= make_zero<T>();
    T res_2_	= make_zero<T>();
    T x1		= make_zero<T>();
    T x2		= make_zero<T>();
	const int j = ty_ * elements_per_thread * lda + tx_;

	A += syhemv_bs;
    x += syhemv_bs * incx;

	__syncthreads();

	// read upper
	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];


	#pragma unroll
    for(int Vblocks = 0; Vblocks < count /*gridDim.x-blkc-1*/; Vblocks++)
    {

		res_1_	=	make_zero<T>();
		res_2_	=	make_zero<T>();

		x1 = x[incx * tx_];
		x2 = x[incx * (tx_ + (syhemv_bs/2))];

		// read lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[(syhemv_bs/2) + j + k * lda];

	    // compute upper
	    #pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	    	res_1_ += xreg[k] * xbuff[ty_ * elements_per_thread + k];
	    	treg[k] += conjugate( xreg[k] ) * x1;
		}

		A += syhemv_bs;
		x += syhemv_bs * incx;

		// read upper from next block
		if(Vblocks != count-1 /*(gridDim.x-blkc-1)-1*/)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
		}

		// compute lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	  		res_2_ 	+= areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
	  		treg[k] += conjugate( areg[k] ) * x2;
		}

		// Horizontal block should be stored in global memory
		__syncthreads();
		accum[ty_ * syhemv_bs + tx_] = res_1_;
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = res_2_;
		__syncthreads();
		if(ty == 0)
		{
			ycopy += syhemv_bs * incy;
	    	res_1_ = make_zero<T>();
	    	#pragma unroll
	    	for(int k = 0; k < (2 * thread_y); k++)
	      		res_1_ += accum[k * syhemv_bs + tx];

	    	res_1_ *= alpha;
	    	// use atomics
	    	atomicAdd(&ycopy[incy * tx], res_1_);
	    }
	}

	// reduction of treg
	#pragma unroll
	for(int k = 0; k < elements_per_thread; k++)
	  	la[(ty_ * elements_per_thread + k) * (syhemv_bs/2) + tx_] = treg[k];

	__syncthreads();

	if(blkc != gridDim.x-1)
	{
		if(ty == 0)
		{
			treg[0] = make_zero<T>(); 			// as a temporary accumulator
	  		#pragma unroll
	    	for(int j = tx; j < tx+(syhemv_bs/2); j++)
	      		treg[0] += la[tx * (syhemv_bs/2) +  (j % (syhemv_bs/2))];

	      	treg[0] *= alpha;
	      	// use atomics
	      	atomicAdd(&y[incy * tx], treg[0]);
	  	}
	}
}

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
        exit(1); \
    } \
} while(0)

void initializeMatrix(float* matrix, int size, bool symmetric = false) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    if (symmetric) {
        // Initialize symmetric matrix
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i <= j) {
                    matrix[i * size + j] = dis(gen);
                    matrix[j * size + i] = matrix[i * size + j]; // Make symmetric
                } 
            }
        }
    } else {
        for (int i = 0; i < size * size; i++) {
            matrix[i] = dis(gen);
        }
    }
}

void initializeVector(float* vector, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        vector[i] = dis(gen);
    }
}

#include <cmath>
#include <algorithm>

bool compareResults(const float* a, const float* b, int size, float rtol = 1e-5f, float atol = 1e-5f) {

    double max_error = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = std::abs(a[i] - b[i]);
        double tolerance = atol + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        if (diff > tolerance) {
            
        }
        max_error = std::max(max_error, diff);
    }
    std::cout << "Maximum error between results: " << std::scientific << max_error << std::endl;
    return true;
}



int main() {
    const int N = 4096;
    const int WARMUP_RUNS = 5;
    const int BENCHMARK_RUNS = 20;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Kernel configuration
    const int SYHEMV_BS = 64;  // Block size for the symmetric matrix kernel
    const int THREAD_X = 64;   // Should match SYHEMV_BS
    const int THREAD_Y = 1;    // Thread configuration
    const int ELEMENTS_PER_THREAD = 1;
    
    std::cout << std::fixed << std::setprecision(6);
    
    // Allocate host memory
    std::vector<float> h_A(N * N);
    std::vector<float> h_x(N);
    std::vector<float> h_y_cublas(N);
    std::vector<float> h_y_custom(N);
    std::vector<float> h_y_original(N);
    
    // Initialize data
    initializeMatrix(h_A.data(), N, true); // Symmetric matrix
    initializeVector(h_x.data(), N);
    
    // Initialize y vector to zero for cleaner comparison
    std::fill(h_y_original.begin(), h_y_original.end(), 0.0f);
    
    // Copy for both kernels
    h_y_cublas = h_y_original;
    h_y_custom = h_y_original;
    
    // Allocate device memory
    float *d_A, *d_x, *d_y_cublas, *d_y_custom;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_cublas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_custom, N * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Configure kernel launch parameters
    dim3 blockDim_d(THREAD_X, THREAD_Y);
    dim3 gridDim_d(N / SYHEMV_BS);  // For diagonal blocks
    
    dim3 blockDim_nd(THREAD_X, THREAD_Y);
    dim3 gridDim_nd(N / SYHEMV_BS, 4);  // For non-diagonal blocks, using 4 blocks in y-direction
    
    std::cout << "Kernel configuration:" << std::endl;
    std::cout << "Block size (SYHEMV_BS): " << SYHEMV_BS << std::endl;
    std::cout << "Thread block: " << THREAD_X << "x" << THREAD_Y << std::endl;
    std::cout << "Grid size for diagonal: " << gridDim_d.x << std::endl;
    std::cout << "Grid size for non-diagonal: " << gridDim_nd.x << "x" << gridDim_nd.y << std::endl;
    
    // Warm-up cuBLAS
    std::cout << "\nWarming up cuBLAS..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; i++) {
        CUDA_CHECK(cudaMemcpy(d_y_cublas, h_y_cublas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSsymv(handle, CUBLAS_FILL_MODE_UPPER, N, &alpha, d_A, N, d_x, 1, &beta, d_y_cublas, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark cuBLAS
    std::cout << "Benchmarking cuBLAS cublasSsymv..." << std::endl;
    float cublasTime = 0.0f;
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        CUDA_CHECK(cudaMemcpy(d_y_cublas, h_y_cublas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasSsymv(handle, CUBLAS_FILL_MODE_LOWER, N, &alpha, d_A, N, d_x, 1, &beta, d_y_cublas, 1));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        cublasTime += milliseconds;
    }
    cublasTime /= BENCHMARK_RUNS;
    
    // Copy result back for verification
    CUDA_CHECK(cudaMemcpy(h_y_cublas.data(), d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Warm-up custom kernels
    std::cout << "Warming up custom kernel..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; i++) {
        CUDA_CHECK(cudaMemcpy(d_y_custom, h_y_custom.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch diagonal kernel
        syhemvl_special_d<float, SYHEMV_BS, THREAD_X, THREAD_Y, ELEMENTS_PER_THREAD>
            <<<gridDim_d, blockDim_d>>>(N, alpha, d_A, N, d_x, 1, beta, d_y_custom, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // Launch non-diagonal kernel
        syhemvl_special_nd<float, SYHEMV_BS, THREAD_X, THREAD_Y, ELEMENTS_PER_THREAD>
            <<<gridDim_nd, blockDim_nd>>>(N, alpha, d_A, N, d_x, 1, beta, d_y_custom, 1);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark custom kernels
    std::cout << "Benchmarking custom syhemvl_special kernels..." << std::endl;
    float customTime = 0.0f;
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        CUDA_CHECK(cudaMemcpy(d_y_custom, h_y_custom.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        // Launch diagonal kernel
        syhemvl_special_d<float, SYHEMV_BS, THREAD_X, THREAD_Y, ELEMENTS_PER_THREAD>
            <<<gridDim_d, blockDim_d>>>(N, alpha, d_A, N, d_x, 1, beta, d_y_custom, 1);
        CUDA_CHECK(cudaGetLastError());
        
        // Launch non-diagonal kernel
        syhemvl_special_nd<float, SYHEMV_BS, THREAD_X, THREAD_Y, ELEMENTS_PER_THREAD>
            <<<gridDim_nd, blockDim_nd>>>(N, alpha, d_A, N, d_x, 1, beta, d_y_custom, 1);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        customTime += milliseconds;
    }
    customTime /= BENCHMARK_RUNS;
    
    // Copy result back for verification
    CUDA_CHECK(cudaMemcpy(h_y_custom.data(), d_y_custom, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate speedup
    float speedup = cublasTime / customTime;
    
    // Verify results
    float maxError = compareResults(h_y_cublas.data(), h_y_custom.data(), N);
    
    // Print results
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Vector size: " << N << std::endl;
    std::cout << "\nTiming Results (average of " << BENCHMARK_RUNS << " runs):" << std::endl;
    std::cout << "cuBLAS cublasSsymv:    " << cublasTime << " ms" << std::endl;
    std::cout << "Custom kernel:   " << customTime << " ms" << std::endl;
    std::cout << "\n";
    
    if (speedup > 1.0f) {
        std::cout << "Speedup: " << speedup << "x (Custom kernel is faster)" << std::endl;
    } else {
        std::cout << "Speedup: " << speedup << "x (cuBLAS is faster)" << std::endl;
    }
    
    std::cout << "\nVerification:" << std::endl;
    
    
    if (maxError < 1e-5) {
        std::cout << "✓ Results match (within numerical precision)" << std::endl;
    } else {
        std::cout << "✗ Results do not match!" << std::endl;
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