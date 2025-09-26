#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

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
__device__ __forceinline__ T make_real(T x) {
    return x; // For double, make_real is identity
}

// Custom SYMV kernel based on the provided template
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvl_generic_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy,
				    int	    n_mod_syhemv_bs)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int blkc = blockIdx.x ;
    const int td  = (thread_x * ty ) + tx;

    T res  = make_zero<T>();
    T yold = make_zero<T>();

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

	// Advance x
	x += (blkc * syhemv_bs) * incx;

	// Advacne y
	y += (blkc * syhemv_bs) * incy;

	// load part of vector x
	if(blkc == gridDim.x-1)
	{
		if(ty == 0)
		{
	  		if(tx < n_mod_syhemv_bs)
	  		{
	    		buff[tx] = x[incx * tx];
	    		yold = beta * y[tx * incy];
	    	}
	    	else
	    	{
	    		buff[tx] = make_zero<T>();
	    		yold = make_zero<T>();
	    	}
	  	}
	}
	else
	{
	  	if(ty == 0)
	  	{
			buff[tx] = x[incx * tx];
			yold = beta * y[tx * incy];
		}
	} // end of load part of vector x

	// init shmem (last TB only)
	if(blkc == gridDim.x-1)
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)
			la[j * syhemv_bs + td ] = make_zero<T>();
		__syncthreads();

		if(tx >= n_mod_syhemv_bs) return; 	// these threads should not read any useful data
	}

	// load a bock of data
	if(blkc == gridDim.x-1)
	{
		int j;
		#pragma unroll
		for(j = 0; j < n_mod_syhemv_bs/thread_y; j++)
			la[(j*thread_y) * syhemv_bs + td] = A[(j*thread_y) * lda];

		if(ty < (n_mod_syhemv_bs%thread_y))
			la[(j*thread_y) * syhemv_bs + td] = A[(j*thread_y) * lda];
	}
	else
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)
			la[j * syhemv_bs + td] = A[j * lda];
	}
	// end of reading a diagonal block of data

	__syncthreads();

	// mirror necessary elements in first chunk
	if(ty > tx)
		la[td] = conjugate( la[tx * syhemv_bs + ty] );
	else
		la[td] = la[td];

	#pragma unroll
	for(int j = thread_y; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx - ty) < j)
			la[tx + (ty + j) * syhemv_bs] = conjugate( la[ty + j + tx * syhemv_bs] );

	// mirror second chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx-ty) < (j + (syhemv_bs/2)))
			la[syhemv_bs * ((syhemv_bs/2) + j + ty) + tx] = conjugate( la[syhemv_bs * tx + (syhemv_bs/2) + j + ty] );

	// ignore imaginary part of diagonal elements
	if(ty == 0) la[tx * syhemv_bs + tx] = make_real(la[tx * syhemv_bs + tx]);

	__syncthreads();

	// compute first chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	// compute second chunk
	#pragma unroll
	for(int j = (syhemv_bs/2); j < 2 * (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	accum[td] = res;
	__syncthreads();
	if(ty == 0)
	{
		res = make_zero<T>();
	  	#pragma unroll
	  	for(int j = 0; j < thread_y; j++)
			res += accum[j * syhemv_bs + tx];
	  	res *= alpha;
	  	res += yold;
	  	if(blkc == gridDim.x-1){if(tx < n_mod_syhemv_bs)y[tx * incy] = res;}
	  	else{y[tx * incy] = res;}
	}
}

template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread >
__global__ void
syhemvl_generic_nd( int n, T alpha,
                               T *A, int lda,
                               T *x, int incx,
                               T  beta,
                               T *y, int incy,
								int     n_mod_syhemv_bs)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int blkc = blockIdx.x ;
    const int by 	= blockIdx.y;
    const int td  = (thread_x * ty ) + tx;
    const int tx_  = td % (syhemv_bs/2);
    const int ty_  = td / (syhemv_bs/2);
    T *xcopy, *ycopy;

    int count = (gridDim.x-blkc-1-1)/gridDim.y;

    T xreg[elements_per_thread];
    T areg[elements_per_thread];
    T treg[elements_per_thread] = {make_zero<T>()};

    T res_1_	= make_zero<T>();
    T res_2_	= make_zero<T>();
    T x1		= make_zero<T>();
    T x2		= make_zero<T>();

    __shared__ T la   [syhemv_bs * (syhemv_bs/2)];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    __shared__ T xbuff[syhemv_bs];

    if(blkc == gridDim.x - 1)return;

    // Advance 'A' to start of diagonal blocks first
    A += syhemv_bs * blkc * (lda + 1);
    // divide work among the y-direction of the grid
	A += (by * count) * syhemv_bs;

    // Advance 'x'
    x += (blkc * syhemv_bs) * incx;
    xcopy = x;
    x += (by * count * syhemv_bs) * incx;

    if(ty == 0) xbuff[tx] = xcopy[incx * tx];

    //Advance 'y'
	y += (blkc * syhemv_bs) * incy;
    ycopy = y;
    ycopy += (by * count * syhemv_bs) * incy;

    if(by == gridDim.y-1) count += ((gridDim.x-blkc-1-1)%gridDim.y) ;//- 1;	// -1 for the generic block at the bottom
    if(by != gridDim.y-1){if(count == 0) return;}

	int j = ty_ * elements_per_thread * lda + tx_;

	__syncthreads();

    A += syhemv_bs;
    x += syhemv_bs * incx;

    if(blkc < gridDim.x-2)		// to prevent out of bound access
    {
    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
			xreg[k] = A[j + k * lda];
    	x1 = x[incx * tx_];
    }

    A -= syhemv_bs;
    x -= syhemv_bs * incx;

    #pragma unroll
    for(int Vblocks = 0; Vblocks < count /*(gridDim.x-blkc-1)-1*/; Vblocks++)
    {
		A += syhemv_bs;
		x += syhemv_bs * incx;

		res_1_ = make_zero<T>();
		res_2_ = make_zero<T>();

		x2 = x[incx * (tx_ + (syhemv_bs/2))];

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[(syhemv_bs/2) + j + k * lda];

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	    	res_1_ += xreg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx* (ty_ * elements_per_thread + k)];
	    	treg[k] += conjugate( xreg[k] ) * x1;
		}

		A += syhemv_bs;
		x += syhemv_bs * incx;

		if(Vblocks != count-1 /*((gridDim.x-blkc-1)-1)-1*/)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
	  		x1 = x[incx * tx_];
	  	}

		A -= syhemv_bs;
		x -= syhemv_bs * incx;

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	  		res_2_ += areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
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
    }// end of for loop on blocks

    //////////////////////////////////////////////////
    // last irregular tile
    if(by == gridDim.y-1)
    {
    	res_1_ = make_zero<T>();
    	res_2_ = make_zero<T>();

		A += syhemv_bs;
		x += syhemv_bs * incx;

    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
    	{
    		xreg[k] = make_zero<T>();
    		areg[k] = make_zero<T>();
    	}

    	if(tx_ < n_mod_syhemv_bs)
    	{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
				xreg[k] = A[j + k * lda];

			x1 = x[incx * tx_];
		}

		if( (tx_+(syhemv_bs/2)) < n_mod_syhemv_bs)
		{
			#pragma unroll
    		for(int k = 0; k < elements_per_thread; k++)
				areg[k] = A[(syhemv_bs/2) + j + k * lda];

			x2 = x[incx * (tx_ + (syhemv_bs/2))];
		}

    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
    	{
			res_1_ 	+= xreg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
			treg[k] += conjugate( xreg[k] ) * x1;
		}

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
			res_2_	+= areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
			treg[k] += conjugate( areg[k] ) * x2;
		}

    	// Horizontal block reduction
		__syncthreads();
		accum[ty_ * syhemv_bs + tx_] = res_1_;
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = res_2_;
		__syncthreads();
		if(ty == 0)
		{
			ycopy += syhemv_bs * incy;
	   		res_1_ = make_zero<T>();;
	   		#pragma unroll
	   		for(int k = 0; k < (2 * thread_y); k++)
	    		res_1_ += accum[k * syhemv_bs + tx];

	    	res_1_ *= alpha;
	    	// use atomics
	    	if(tx < n_mod_syhemv_bs)atomicAdd(&ycopy[incy * tx], res_1_);
		}
	}

	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
    	la[(ty_ * elements_per_thread + k) * (syhemv_bs/2) + tx_] = treg[k];

    __syncthreads();		// important

    if(ty == 0)
    {
		treg[0] = make_zero<T>(); // tmp accumulator
		#pragma unroll
		for(int j = tx; j < tx+(syhemv_bs/2); j++)
	  		treg[0] += la[tx * (syhemv_bs/2) +  (j % (syhemv_bs/2))];

	   	treg[0] *= alpha;
	   	atomicAdd(&y[incy * tx], treg[0]);
	}
}

// Error checking function equivalent to torch.allclose
bool compareResults(const double* a, const double* b, int size, double rtol = 1e-5, double atol = 1e-8) {
    double max_error = 0.0;
    int max_error_index = -1;
    int first_mismatch_index = -1;
    int mismatch_count = 0;
    
    for (int i = 0; i < size; i++) {
        double diff = std::abs(a[i] - b[i]);
        double tolerance = atol + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        
        if (diff > max_error) {
            max_error = diff;
            max_error_index = i;
        }
        
        if (diff > tolerance) {
            if (first_mismatch_index == -1) {
                first_mismatch_index = i;
            }
            mismatch_count++;
            
            // Print first few mismatches for debugging
            if (mismatch_count <= 5) {
                std::cout << "Mismatch at index " << i << ": " 
                         << "a[" << i << "] = " << std::scientific << a[i] 
                         << ", b[" << i << "] = " << std::scientific << b[i] 
                         << ", diff = " << std::scientific << diff 
                         << ", tolerance = " << std::scientific << tolerance << std::endl;
            }
        }
    }
    
    std::cout << "Maximum error between results: " << std::scientific << max_error;
    if (max_error_index != -1) {
        std::cout << " (at index " << max_error_index << ")";
    }
    std::cout << std::endl;
    
    if (mismatch_count > 0) {
        std::cout << "Total mismatches: " << mismatch_count << " out of " << size << " elements" << std::endl;
        if (mismatch_count > 5) {
            std::cout << "(Only showing first 5 mismatches)" << std::endl;
        }
        return false;
    }
    
    return true;
}

void initializeSymmetricMatrix(double* matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Initialize matrix with random values and ensure symmetry
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i <= j) {
                matrix[i * n + j] = dist(gen);
                matrix[j * n + i] = matrix[i * n + j]; // Ensure symmetry
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

// Define constants following the driver pattern
const int dsymv_lower_bs = 64;
const int dsymv_lower_ty = 4;
const int dsymv_lower_by = 4;

// Wrapper function for custom kernel launch following the driver pattern
void launchCustomSymv(int n, double alpha, double *d_A, int lda, double *d_x, int incx,
                     double beta, double *d_y, int incy) {
    char uplo = 'L'; // Lower triangular
    
    // Handle negative increments (following driver pattern)
    if(incx < 0) d_x -= (n-1) * incx;
    if(incy < 0) d_y -= (n-1) * incy;
    
    // Configuration params
    const int dsymv_bs = dsymv_lower_bs;
    const int thread_x = dsymv_bs;
    const int thread_y = dsymv_lower_ty;
    const int elements_per_thread = (dsymv_bs/(2*thread_y));
    
    int mod = n % dsymv_bs;
    int blocks = n / dsymv_bs + (mod != 0);
    dim3 dimBlock(thread_x, thread_y);
    dim3 dimGrid(blocks, 1);
    dim3 dimGrid_(blocks, dsymv_lower_by);
    
    if(mod == 0) {
        // For exact multiples, we would need special kernels
        // For now, use generic kernels with mod = 0
        syhemvl_generic_d<double, dsymv_bs, thread_x, thread_y, elements_per_thread>
            <<<dimGrid, dimBlock>>>(n, alpha, d_A, lda, d_x, incx, beta, d_y, incy, 0);
        syhemvl_generic_nd<double, dsymv_bs, thread_x, thread_y, elements_per_thread>
            <<<dimGrid_, dimBlock>>>(n, alpha, d_A, lda, d_x, incx, beta, d_y, incy, 0);
    } else {
        // Generic case for irregular dimensions
        syhemvl_generic_d<double, dsymv_bs, thread_x, thread_y, elements_per_thread>
            <<<dimGrid, dimBlock>>>(n, alpha, d_A, lda, d_x, incx, beta, d_y, incy, mod);
        syhemvl_generic_nd<double, dsymv_bs, thread_x, thread_y, elements_per_thread>
            <<<dimGrid_, dimBlock>>>(n, alpha, d_A, lda, d_x, incx, beta, d_y, incy, mod);
    }
}

int main() {
    const int N = 4099;
    const double alpha = 1.0;
    const double beta = 0.0;
    const int incx = 1;
    const int incy = 1;
    const int warmup_runs = 5;
    const int benchmark_runs = 20;
    
    // Host memory allocation
    std::vector<double> h_A(N * N);
    std::vector<double> h_x(N);
    std::vector<double> h_y_cublas(N);
    std::vector<double> h_y_custom(N);
    std::vector<double> h_y_original(N);
    
    // Initialize matrices and vectors
    initializeSymmetricMatrix(h_A.data(), N);
    initializeVector(h_x.data(), N);
    initializeVector(h_y_original.data(), N);
    
    // Copy original y for both tests
    std::copy(h_y_original.begin(), h_y_original.end(), h_y_cublas.begin());
    std::copy(h_y_original.begin(), h_y_original.end(), h_y_custom.begin());
    
    // Device memory allocation
    double *d_A, *d_x, *d_y_cublas, *d_y_custom;
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_cublas, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y_custom, N * sizeof(double)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_cublas, h_y_cublas.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_custom, h_y_custom.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
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
        CHECK_CUBLAS(cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, N, &alpha, d_A, N, d_x, incx, &beta, d_y_cublas, incy));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark cuBLAS
    std::cout << "Benchmarking cuBLAS dsymv..." << std::endl;
    float cublas_total_time = 0.0f;
    
    for (int i = 0; i < benchmark_runs; i++) {
        // Reset y vector
        CHECK_CUDA(cudaMemcpy(d_y_cublas, h_y_original.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, N, &alpha, d_A, N, d_x, incx, &beta, d_y_cublas, incy));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        cublas_total_time += milliseconds;
    }
    float cublas_avg_time = cublas_total_time / benchmark_runs;
    
    // Copy cuBLAS result back to host for verification
    CHECK_CUDA(cudaMemcpy(h_y_cublas.data(), d_y_cublas, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Warm-up custom kernel
    std::cout << "Warming up custom kernel..." << std::endl;
    for (int i = 0; i < warmup_runs; i++) {
        launchCustomSymv(N, alpha, d_A, N, d_x, incx, beta, d_y_custom, incy);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark custom kernel
    std::cout << "Benchmarking custom symv kernel..." << std::endl;
    float custom_total_time = 0.0f;
    
    for (int i = 0; i < benchmark_runs; i++) {
        // Reset y vector
        CHECK_CUDA(cudaMemcpy(d_y_custom, h_y_original.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaEventRecord(start));
        launchCustomSymv(N, alpha, d_A, N, d_x, incx, beta, d_y_custom, incy);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        custom_total_time += milliseconds;
    }
    float custom_avg_time = custom_total_time / benchmark_runs;
    
    // Copy custom kernel result back to host for verification
    CHECK_CUDA(cudaMemcpy(h_y_custom.data(), d_y_custom, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Debug: Print first few elements for comparison
    std::cout << "\nDebug - First 10 elements comparison:" << std::endl;
    std::cout << "Index\tcuBLAS\t\tCustom\t\tDiff" << std::endl;
    for (int i = 0; i < 10; i++) {
        double diff = std::abs(h_y_cublas[i] - h_y_custom[i]);
        std::cout << i << "\t" << std::scientific << std::setprecision(6) 
                  << h_y_cublas[i] << "\t" << h_y_custom[i] << "\t" << diff << std::endl;
    }
    
    // Performance results
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << std::endl;
    
    std::cout << "Timing Results (average of " << benchmark_runs << " runs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "cuBLAS dsymv:     " << cublas_avg_time << " ms" << std::endl;
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
    bool results_match = compareResults(h_y_cublas.data(), h_y_custom.data(), N);
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
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y_cublas));
    CHECK_CUDA(cudaFree(d_y_custom));
    
    return 0;
}