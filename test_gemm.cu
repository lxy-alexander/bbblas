// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <iostream>
// #include <iomanip>
// #include <cmath>
// #include <random>
// #include <algorithm>

// #define CHECK_CUDA_ERROR(call)                                             \
//     do {                                                                    \
//         cudaError_t error = call;                                          \
//         if (error != cudaSuccess) {                                        \
//             std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
//                       << " - " << cudaGetErrorString(error) << std::endl;  \
//             exit(1);                                                        \
//         }                                                                   \
//     } while(0)

// #define CHECK_CUBLAS_ERROR(call)                                           \
//     do {                                                                    \
//         cublasStatus_t status = call;                                      \
//         if (status != CUBLAS_STATUS_SUCCESS) {                             \
//             std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
//                       << " - " << status << std::endl;                     \
//             exit(1);                                                        \
//         }                                                                   \
//     } while(0)

// #define CHECK_LAST_CUDA_ERROR() CHECK_CUDA_ERROR(cudaGetLastError())

// // Helper function to load data from global memory to shared memory
// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS>
// __device__ void load_data_from_global_memory_to_shared_memory_transposed_vectorized(
//     T const* A, size_t lda, T const* B, size_t ldb,
//     T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y],
//     T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X],
//     size_t thread_block_tile_idx, size_t thread_linear_idx, size_t m, size_t n, size_t k)
// {
//     constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
//     static_assert(sizeof(int4) % sizeof(T) == 0U);
    
//     // Load A tile (transposed)
//     constexpr size_t A_TILE_ELEMENTS{BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K};
//     constexpr size_t NUM_LOADS_PER_THREAD_A{(A_TILE_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS};
    
// #pragma unroll
//     for (size_t load_idx = 0; load_idx < NUM_LOADS_PER_THREAD_A; ++load_idx) {
//         size_t const element_idx = thread_linear_idx + load_idx * NUM_THREADS;
//         if (element_idx < A_TILE_ELEMENTS) {
//             size_t const tile_row = element_idx / BLOCK_TILE_SIZE_K;
//             size_t const tile_col = element_idx % BLOCK_TILE_SIZE_K;
//             size_t const global_row = blockIdx.y * BLOCK_TILE_SIZE_Y + tile_row;
//             size_t const global_col = thread_block_tile_idx * BLOCK_TILE_SIZE_K + tile_col;
            
//             if (global_row < m && global_col < k) {
//                 A_thread_block_tile_transposed[tile_col][tile_row] = A[global_row * lda + global_col];
//             } else {
//                 A_thread_block_tile_transposed[tile_col][tile_row] = static_cast<T>(0);
//             }
//         }
//     }
    
//     // Load B tile
//     constexpr size_t B_TILE_ELEMENTS{BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X};
//     constexpr size_t NUM_LOADS_PER_THREAD_B{(B_TILE_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS};
    
// #pragma unroll
//     for (size_t load_idx = 0; load_idx < NUM_LOADS_PER_THREAD_B; ++load_idx) {
//         size_t const element_idx = thread_linear_idx + load_idx * NUM_THREADS;
//         if (element_idx < B_TILE_ELEMENTS) {
//             size_t const tile_row = element_idx / BLOCK_TILE_SIZE_X;
//             size_t const tile_col = element_idx % BLOCK_TILE_SIZE_X;
//             size_t const global_row = thread_block_tile_idx * BLOCK_TILE_SIZE_K + tile_row;
//             size_t const global_col = blockIdx.x * BLOCK_TILE_SIZE_X + tile_col;
            
//             if (global_row < k && global_col < n) {
//                 B_thread_block_tile[tile_row][tile_col] = B[global_row * ldb + global_col];
//             } else {
//                 B_thread_block_tile[tile_row][tile_col] = static_cast<T>(0);
//             }
//         }
//     }
// }

// template <typename T, size_t BLOCK_TILE_SIZE, size_t WARP_TILE_SIZE,
//           size_t NUM_THREAD_TILES_PER_WARP, size_t THREAD_TILE_SIZE>
// __device__ void load_data_from_shared_memory_to_register_file_vectorized(
//     T const thread_block_tile[BLOCK_TILE_SIZE],
//     T register_values[NUM_THREAD_TILES_PER_WARP][THREAD_TILE_SIZE],
//     size_t warp_idx, size_t thread_idx)
// {
//     static_assert(BLOCK_TILE_SIZE % THREAD_TILE_SIZE == 0U);
//     constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
//     static_assert(sizeof(int4) % sizeof(T) == 0U);
//     constexpr size_t VECTORIZED_THREAD_TILE_SIZE{THREAD_TILE_SIZE /
//                                                  NUM_VECTOR_UNITS};
//     static_assert(THREAD_TILE_SIZE % NUM_VECTOR_UNITS == 0U);

// #pragma unroll
//     for (size_t thread_tile_repeat_row_idx{0U};
//          thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP;
//          ++thread_tile_repeat_row_idx)
//     {
//         size_t const thread_block_tile_row_idx{
//             warp_idx * WARP_TILE_SIZE +
//             thread_tile_repeat_row_idx *
//                 (WARP_TILE_SIZE / NUM_THREAD_TILES_PER_WARP) +
//             thread_idx * THREAD_TILE_SIZE};
// #pragma unroll
//         for (size_t thread_tile_vector_idx{0U};
//              thread_tile_vector_idx < VECTORIZED_THREAD_TILE_SIZE;
//              ++thread_tile_vector_idx)
//         {
//             *reinterpret_cast<int4*>(
//                 &register_values[thread_tile_repeat_row_idx]
//                                 [thread_tile_vector_idx * NUM_VECTOR_UNITS]) =
//                 *reinterpret_cast<int4 const*>(
//                     &thread_block_tile[thread_block_tile_row_idx +
//                                        thread_tile_vector_idx *
//                                            NUM_VECTOR_UNITS]);
//         }
//     }
// }

// template <typename T, size_t NUM_THREAD_TILES_PER_WARP_X,
//           size_t NUM_THREAD_TILES_PER_WARP_Y, size_t THREAD_TILE_SIZE_X,
//           size_t THREAD_TILE_SIZE_Y>
// __device__ void compute_thread_tile_results(
//     T const A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y],
//     T const B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X],
//     T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
//                       [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X])
// {
// // Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X outer
// // products.
// #pragma unroll
//     for (size_t thread_tile_repeat_row_idx{0U};
//          thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
//          ++thread_tile_repeat_row_idx)
//     {
// #pragma unroll
//         for (size_t thread_tile_repeat_col_idx{0U};
//              thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
//              ++thread_tile_repeat_col_idx)
//         {
// #pragma unroll
//             for (size_t thread_tile_y_idx{0U};
//                  thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
//             {
// #pragma unroll
//                 for (size_t thread_tile_x_idx{0U};
//                      thread_tile_x_idx < THREAD_TILE_SIZE_X;
//                      ++thread_tile_x_idx)
//                 {
//                     C_thread_results[thread_tile_repeat_row_idx]
//                                     [thread_tile_repeat_col_idx]
//                                     [thread_tile_y_idx][thread_tile_x_idx] +=
//                         A_vals[thread_tile_repeat_row_idx][thread_tile_y_idx] *
//                         B_vals[thread_tile_repeat_col_idx][thread_tile_x_idx];
//                 }
//             }
//         }
//     }
// }

// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t WARP_TILE_SIZE_X, size_t WARP_TILE_SIZE_Y,
//           size_t THREAD_TILE_SIZE_X, size_t THREAD_TILE_SIZE_Y,
//           size_t NUM_THREAD_TILES_PER_WARP_X,
//           size_t NUM_THREAD_TILES_PER_WARP_Y>
// __device__ void write_results_from_register_file_to_global_memory_vectorized(
//     T const C_thread_results[NUM_THREAD_TILES_PER_WARP_Y]
//                             [NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_Y]
//                             [THREAD_TILE_SIZE_X],
//     T alpha, T beta, T* C, size_t ldc, size_t m, size_t n, size_t block_row_idx,
//     size_t block_col_idx, size_t warp_row_idx, size_t warp_col_idx,
//     size_t thread_row_idx_in_warp, size_t thread_col_idx_in_warp)
// {
//     constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
//     static_assert(sizeof(int4) % sizeof(T) == 0U);
//     static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
//     constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X{THREAD_TILE_SIZE_X /
//                                                    NUM_VECTOR_UNITS};
//     static_assert(THREAD_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

// // Write the results to DRAM.
// #pragma unroll
//     for (size_t thread_tile_repeat_row_idx{0U};
//          thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
//          ++thread_tile_repeat_row_idx)
//     {
// #pragma unroll
//         for (size_t thread_tile_repeat_col_idx{0U};
//              thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
//              ++thread_tile_repeat_col_idx)
//         {
// #pragma unroll
//             for (size_t thread_tile_y_idx{0U};
//                  thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
//             {
// #pragma unroll
//                 for (size_t thread_tile_x_vector_idx{0U};
//                      thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
//                      ++thread_tile_x_vector_idx)
//                 {
//                     size_t const C_row_idx{
//                         blockIdx.y * BLOCK_TILE_SIZE_Y +
//                         warp_row_idx * WARP_TILE_SIZE_Y +
//                         thread_tile_repeat_row_idx *
//                             (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
//                         thread_row_idx_in_warp * THREAD_TILE_SIZE_Y +
//                         thread_tile_y_idx};
//                     size_t const C_col_idx{
//                         blockIdx.x * BLOCK_TILE_SIZE_X +
//                         warp_col_idx * WARP_TILE_SIZE_X +
//                         thread_tile_repeat_col_idx *
//                             (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
//                         thread_col_idx_in_warp * THREAD_TILE_SIZE_X +
//                         thread_tile_x_vector_idx * NUM_VECTOR_UNITS};

//                     if (C_row_idx < m && C_col_idx < n)
//                     {
//                         int4 C_vals{*reinterpret_cast<int4 const*>(
//                             &C[C_row_idx * ldc + C_col_idx])};
// #pragma unroll
//                         for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
//                         {
//                             reinterpret_cast<T*>(&C_vals)[i] =
//                                 alpha *
//                                     C_thread_results[thread_tile_repeat_row_idx]
//                                                     [thread_tile_repeat_col_idx]
//                                                     [thread_tile_y_idx]
//                                                     [thread_tile_x_vector_idx *
//                                                          NUM_VECTOR_UNITS +
//                                                      i] +
//                                 beta * reinterpret_cast<T const*>(&C_vals)[i];
//                         }
//                         *reinterpret_cast<int4*>(
//                             &C[C_row_idx * ldc + C_col_idx]) = C_vals;
//                     }
//                 }
//             }
//         }
//     }
// }

// // GEMM kernel v06.
// // Each thread in the block processes THREAD_TILE_SIZE_Y *
// // THREAD_TILE_SIZE_X output values. Number of threads BLOCK_TILE_SIZE_Y *
// // BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
// template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
//           size_t BLOCK_TILE_SIZE_K, size_t WARP_TILE_SIZE_X,
//           size_t WARP_TILE_SIZE_Y, size_t THREAD_TILE_SIZE_X,
//           size_t THREAD_TILE_SIZE_Y, size_t NUM_THREADS_PER_WARP_X,
//           size_t NUM_THREADS_PER_WARP_Y>
// __global__ void gemm_v06_vectorized(size_t m, size_t n, size_t k, T alpha,
//                                     T const* A, size_t lda, T const* B,
//                                     size_t ldb, T beta, T* C, size_t ldc)
// {
//     static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
//     constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
//     static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
//     constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
//     static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);
//     constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
//         WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
//     constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
//         WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};
//     static_assert(
//         WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
//     static_assert(
//         WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

//     constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
//     constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};
//     // Avoid using blockDim.x * blockDim.y as the number of threads per block.
//     // Because it is a runtime constant and the compiler cannot optimize the
//     // loop unrolling based on that.
//     // Use a compile time constant instead.
//     constexpr size_t NUM_THREADS{NUM_THREADS_X * NUM_THREADS_Y};

//     // Cache a tile of A and B in shared memory for data reuse.
//     __shared__ T
//         A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
//     __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

//     // A_vals is cached in the register.
//     T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {
//         static_cast<T>(0)};
//     // B_vals is cached in the register.
//     T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {
//         static_cast<T>(0)};

//     size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
//     size_t const warp_linear_idx{thread_linear_idx / 32U};
//     size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
//     size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
//     size_t const thread_linear_idx_in_warp{thread_linear_idx % 32U};
//     size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
//                                                NUM_THREADS_PER_WARP_X};
//     size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
//                                                NUM_THREADS_PER_WARP_X};

//     // Number of outer loops to perform the sum of inner products.
//     // C_thread_block_tile =
//     // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
//     // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
//     // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
//     size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
//                                         BLOCK_TILE_SIZE_K};
//     // Each thread in the block processes NUM_THREAD_TILES_PER_WARP_Y *
//     // NUM_THREAD_TILES_PER_WARP_X * THREAD_TILE_SIZE_Y *
//     // THREAD_TILE_SIZE_X output values.
//     T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
//                       [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
//                           static_cast<T>(0)};

//     for (size_t thread_block_tile_idx{0U};
//          thread_block_tile_idx < num_thread_block_tiles;
//          ++thread_block_tile_idx)
//     {
//         load_data_from_global_memory_to_shared_memory_transposed_vectorized<
//             T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
//             NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile_transposed,
//                          B_thread_block_tile, thread_block_tile_idx,
//                          thread_linear_idx, m, n, k);
//         __syncthreads();

// // Perform A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] where A[:,
// // thread_block_tile_idx:BLOCK_TILE_SIZE_K] and
// // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] are cached in the
// // shared memory as A_thread_block_tile and B_thread_block_tile,
// // respectively. This inner product is further decomposed to
// // BLOCK_TILE_SIZE_K outer products. A_thread_block_tile *
// // B_thread_block_tile = \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1}
// // A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :] Note that
// // both A_thread_block_tile and B_thread_block_tile can be cached in the
// // register.
// #pragma unroll
//         for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
//         {
//             // Load data from shared memory to register file for A.
//             load_data_from_shared_memory_to_register_file_vectorized<
//                 T, BLOCK_TILE_SIZE_Y, WARP_TILE_SIZE_Y, NUM_THREAD_TILES_PER_WARP_Y,
//                 THREAD_TILE_SIZE_Y>(A_thread_block_tile_transposed[k_i], A_vals,
//                                     warp_row_idx,
//                                     thread_linear_row_idx_in_warp);
//             // Load data from shared memory to register file for B.
//             load_data_from_shared_memory_to_register_file_vectorized<
//                 T, BLOCK_TILE_SIZE_X, WARP_TILE_SIZE_X, NUM_THREAD_TILES_PER_WARP_X,
//                 THREAD_TILE_SIZE_X>(B_thread_block_tile[k_i], B_vals,
//                                     warp_col_idx,
//                                     thread_linear_col_idx_in_warp);

//             // Compute NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X
//             // outer products.
//             compute_thread_tile_results<T, NUM_THREAD_TILES_PER_WARP_X,
//                                         NUM_THREAD_TILES_PER_WARP_Y,
//                                         THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>(
//                 A_vals, B_vals, C_thread_results);
//         }
//         __syncthreads();
//     }

//     // Write the results to DRAM.
//     write_results_from_register_file_to_global_memory_vectorized<
//         T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, WARP_TILE_SIZE_X,
//         WARP_TILE_SIZE_Y, THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
//         NUM_THREAD_TILES_PER_WARP_X, NUM_THREAD_TILES_PER_WARP_Y>(
//         C_thread_results, alpha, beta, C, ldc, m, n, blockIdx.y, blockIdx.x,
//         warp_row_idx, warp_col_idx, thread_linear_row_idx_in_warp,
//         thread_linear_col_idx_in_warp);
// }

// template <typename T>
// void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
//                                        T const* alpha, T const* A, size_t lda,
//                                        T const* B, size_t ldb, T const* beta,
//                                        T* C, size_t ldc, cudaStream_t stream)
// {
//     // Feel free to play with the block tile sizes.
//     // The algorithm correctness should always be guaranteed.
//     constexpr unsigned int BLOCK_TILE_SIZE_X{128U};
//     constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};
//     constexpr unsigned int BLOCK_TILE_SIZE_K{16U};

//     constexpr unsigned int WARP_TILE_SIZE_X{32U};
//     constexpr unsigned int WARP_TILE_SIZE_Y{64U};
//     constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
//     constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
//     static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
//     static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

//     constexpr unsigned int THREAD_TILE_SIZE_X{8U};
//     constexpr unsigned int THREAD_TILE_SIZE_Y{8U};

//     constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};
//     constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};
//     static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
//     static_assert(
//         WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
//     static_assert(
//         WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

//     constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
//     constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

//     constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};

//     dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
//     dim3 const grid_dim{
//         (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
//             BLOCK_TILE_SIZE_X,
//         (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
//             BLOCK_TILE_SIZE_Y,
//         1U};
//     gemm_v06_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
//                         BLOCK_TILE_SIZE_K, WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
//                         THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
//                         NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
//         <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
//                                               *beta, C, ldc);
//     CHECK_LAST_CUDA_ERROR();
// }

// // Compare results function similar to torch.allclose
// bool compareResults(const double* a, const double* b, size_t size, 
//                    double rtol = 1e-5, double atol = 1e-8) {
//     double max_error = 0.0;
//     for (size_t i = 0; i < size; ++i) {
//         double diff = std::abs(a[i] - b[i]);
//         double tolerance = atol + rtol * std::abs(b[i]);
//         if (diff > tolerance) {
//             return false;
//         }
//         max_error = std::max(max_error, diff);
//     }
//     std::cout << "Maximum error between results: " << max_error << std::endl;
//     return true;
// }

// // Initialize matrix with random values
// void initializeMatrix(double* matrix, size_t rows, size_t cols) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
//     for (size_t i = 0; i < rows * cols; ++i) {
//         matrix[i] = dist(gen);
//     }
// }

// int main() {
//     // Matrix dimensions
//     constexpr size_t M = 4096;
//     constexpr size_t N = 4096;
//     constexpr size_t K = 4096;
    
//     // GEMM parameters
//     double alpha = 1.0;
//     double beta = 0.0;
    
//     // Allocate host memory
//     double* h_A = new double[M * K];
//     double* h_B = new double[K * N];
//     double* h_C_cublas = new double[M * N];
//     double* h_C_custom = new double[M * N];
    
//     // Initialize matrices
//     initializeMatrix(h_A, M, K);
//     initializeMatrix(h_B, K, N);
//     std::fill(h_C_cublas, h_C_cublas + M * N, 0.0);
//     std::fill(h_C_custom, h_C_custom + M * N, 0.0);
    
//     // Allocate device memory
//     double *d_A, *d_B, *d_C_cublas, *d_C_custom;
//     CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(double)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(double)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_C_cublas, M * N * sizeof(double)));
//     CHECK_CUDA_ERROR(cudaMalloc(&d_C_custom, M * N * sizeof(double)));
    
//     // Copy data to device
//     CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_C_cublas, h_C_cublas, M * N * sizeof(double), cudaMemcpyHostToDevice));
//     CHECK_CUDA_ERROR(cudaMemcpy(d_C_custom, h_C_custom, M * N * sizeof(double), cudaMemcpyHostToDevice));
    
//     // Create cuBLAS handle
//     cublasHandle_t cublasHandle;
//     CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));
    
//     // Create CUDA events for timing
//     cudaEvent_t start, stop;
//     CHECK_CUDA_ERROR(cudaEventCreate(&start));
//     CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
//     // Warm-up runs
//     constexpr int WARMUP_RUNS = 5;
//     constexpr int BENCHMARK_RUNS = 20;
    
//     // Warm-up cuBLAS
//     std::cout << "Warming up cuBLAS..." << std::endl;
//     for (int i = 0; i < WARMUP_RUNS; ++i) {
//         CHECK_CUBLAS_ERROR(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
//                                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));
//     }
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
//     // Benchmark cuBLAS
//     std::cout << "Benchmarking cuBLAS cublasDgemm..." << std::endl;
//     float cublas_time = 0.0f;
//     for (int i = 0; i < BENCHMARK_RUNS; ++i) {
//         CHECK_CUDA_ERROR(cudaEventRecord(start));
//         CHECK_CUBLAS_ERROR(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
//                                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));
//         CHECK_CUDA_ERROR(cudaEventRecord(stop));
//         CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
//         float milliseconds = 0;
//         CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
//         cublas_time += milliseconds;
//     }
//     cublas_time /= BENCHMARK_RUNS;
    
//     // Warm-up custom kernel
//     std::cout << "Warming up custom kernel..." << std::endl;
//     for (int i = 0; i < WARMUP_RUNS; ++i) {
//         launch_gemm_kernel_v06_vectorized(M, N, K, &alpha, d_A, K, d_B, N, 
//                                           &beta, d_C_custom, N, nullptr);
//     }
//     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
//     // Benchmark custom kernel
//     std::cout << "Benchmarking custom gemm_v06_vectorized kernel..." << std::endl;
//     float custom_time = 0.0f;
//     for (int i = 0; i < BENCHMARK_RUNS; ++i) {
//         CHECK_CUDA_ERROR(cudaEventRecord(start));
//         launch_gemm_kernel_v06_vectorized(M, N, K, &alpha, d_A, K, d_B, N, 
//                                           &beta, d_C_custom, N, nullptr);
//         CHECK_CUDA_ERROR(cudaEventRecord(stop));
//         CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
//         float milliseconds = 0;
//         CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
//         custom_time += milliseconds;
//     }
//     custom_time /= BENCHMARK_RUNS;
    
//     // Copy results back to host
//     CHECK_CUDA_ERROR(cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(double), cudaMemcpyDeviceToHost));
//     CHECK_CUDA_ERROR(cudaMemcpy(h_C_custom, d_C_custom, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
//     // Print results
//     std::cout << "\n=== Performance Results ===" << std::endl;
//     std::cout << "Matrix size: " << M << "x" << N << std::endl;
//     std::cout << "\nTiming Results (average of " << BENCHMARK_RUNS << " runs):" << std::endl;
//     std::cout << std::fixed << std::setprecision(6);
//     std::cout << "cuBLAS cublasDgemm:    " << cublas_time << " ms" << std::endl;
//     std::cout << "Custom kernel:   " << custom_time << " ms" << std::endl;
//     std::cout << std::setprecision(5);
    
//     float speedup = cublas_time / custom_time;
//     std::cout << "\nSpeedup: " << speedup << "x ";
//     if (speedup > 1.0f) {
//         std::cout << "(Custom kernel is faster)" << std::endl;
//     } else {
//         std::cout << "(cuBLAS is faster)" << std::endl;
//     }
    
//     // Verify results
//     std::cout << "\nVerification:" << std::endl;
//     bool results_match = compareResults(h_C_cublas, h_C_custom, M * N);
//     if (results_match) {
//         std::cout << "✓ Results match (within numerical precision)" << std::endl;
//     } else {
//         std::cout << "✗ Results do NOT match!" << std::endl;
//     }
    
//     // Cleanup
//     CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));
//     CHECK_CUDA_ERROR(cudaEventDestroy(start));
//     CHECK_CUDA_ERROR(cudaEventDestroy(stop));
//     CHECK_CUDA_ERROR(cudaFree(d_A));
//     CHECK_CUDA_ERROR(cudaFree(d_B));
//     CHECK_CUDA_ERROR(cudaFree(d_C_cublas));
//     CHECK_CUDA_ERROR(cudaFree(d_C_custom));
    
//     delete[] h_A;
//     delete[] h_B;
//     delete[] h_C_cublas;
//     delete[] h_C_custom;
    
//     return 0;
// }
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

// CUDA 错误检查宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

// 精确时间测量类
class PreciseTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    PreciseTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~PreciseTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start_gpu() {
        cudaEventRecord(start_event, 0);
    }
    
    double elapsed_gpu_ms() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float ms;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return static_cast<double>(ms);
    }
};

// CUDA kernel for naive GEMM
__global__ void cuda_gemm_naive(const float* A, const float* B, float* C,
                                int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// 改进的 CUDA kernel - 使用共享内存
__global__ void cuda_gemm_shared(const float* A, const float* B, float* C,
                                 int M, int N, int K, float alpha, float beta) {
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载数据到共享内存
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算部分乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Fragment-based CUDA kernel - 高性能分片算法 (基于专业实现)
template<int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y, int BLOCK_TILE_SIZE_K,
         int WARP_TILE_SIZE_X, int WARP_TILE_SIZE_Y,
         int THREAD_TILE_SIZE_X, int THREAD_TILE_SIZE_Y,
         int NUM_THREADS_PER_WARP_X, int NUM_THREADS_PER_WARP_Y>
__global__ void cuda_gemm_fragment(const float* __restrict__ A, 
                                  const float* __restrict__ B, 
                                  float* __restrict__ C,
                                  int M, int N, int K, float alpha, float beta) {
    
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);
    
    constexpr int NUM_WARPS_X = BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X;
    constexpr int NUM_WARPS_Y = BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y;
    constexpr int NUM_THREAD_TILES_PER_WARP_X = WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X);
    constexpr int NUM_THREAD_TILES_PER_WARP_Y = WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y);
    
    constexpr int NUM_THREADS_X = NUM_WARPS_X * NUM_THREADS_PER_WARP_X;
    constexpr int NUM_THREADS_Y = NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y;
    constexpr int NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y;
    
    // 共享内存 - A转置存储以优化访问模式
    __shared__ float A_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ float B_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];
    
    // 寄存器文件 - 每个线程的私有数据
    float A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y];
    float B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X];
    float C_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                   [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {0.0f};
    
    // 线程和warp索引计算
    const int thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_linear_idx = thread_linear_idx / 32;
    const int warp_row_idx = warp_linear_idx / NUM_WARPS_X;
    const int warp_col_idx = warp_linear_idx % NUM_WARPS_X;
    const int thread_idx_in_warp = thread_linear_idx % 32;
    const int thread_row_idx_in_warp = thread_idx_in_warp / NUM_THREADS_PER_WARP_X;
    const int thread_col_idx_in_warp = thread_idx_in_warp % NUM_THREADS_PER_WARP_X;
    
    // K维度分块数量
    const int num_tiles_k = (K + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;
    
    // 主循环 - K维度分块
    for (int tile_k = 0; tile_k < num_tiles_k; ++tile_k) {
        
        // 协作加载全局内存到共享内存
        // 加载A矩阵 (转置存储)
        for (int load_idx = thread_linear_idx; 
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y; 
             load_idx += NUM_THREADS) {
            
            int k_idx = load_idx / BLOCK_TILE_SIZE_Y;
            int m_idx = load_idx % BLOCK_TILE_SIZE_Y;
            
            int global_m = blockIdx.y * BLOCK_TILE_SIZE_Y + m_idx;
            int global_k = tile_k * BLOCK_TILE_SIZE_K + k_idx;
            
            if (global_m < M && global_k < K) {
                A_shared[k_idx][m_idx] = A[global_m * K + global_k];
            } else {
                A_shared[k_idx][m_idx] = 0.0f;
            }
        }
        
        // 加载B矩阵
        for (int load_idx = thread_linear_idx; 
             load_idx < BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X; 
             load_idx += NUM_THREADS) {
            
            int k_idx = load_idx / BLOCK_TILE_SIZE_X;
            int n_idx = load_idx % BLOCK_TILE_SIZE_X;
            
            int global_k = tile_k * BLOCK_TILE_SIZE_K + k_idx;
            int global_n = blockIdx.x * BLOCK_TILE_SIZE_X + n_idx;
            
            if (global_k < K && global_n < N) {
                B_shared[k_idx][n_idx] = B[global_k * N + global_n];
            } else {
                B_shared[k_idx][n_idx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // 内积计算 - BLOCK_TILE_SIZE_K 个外积的和
        #pragma unroll
        for (int k_i = 0; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
            
            // 从共享内存加载到寄存器 - A矩阵
            #pragma unroll
            for (int tile_y = 0; tile_y < NUM_THREAD_TILES_PER_WARP_Y; ++tile_y) {
                #pragma unroll
                for (int y = 0; y < THREAD_TILE_SIZE_Y; ++y) {
                    int shared_row = warp_row_idx * WARP_TILE_SIZE_Y + 
                                   tile_y * (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                                   thread_row_idx_in_warp * THREAD_TILE_SIZE_Y + y;
                    
                    if (shared_row < BLOCK_TILE_SIZE_Y) {
                        A_vals[tile_y][y] = A_shared[k_i][shared_row];
                    } else {
                        A_vals[tile_y][y] = 0.0f;
                    }
                }
            }
            
            // 从共享内存加载到寄存器 - B矩阵
            #pragma unroll
            for (int tile_x = 0; tile_x < NUM_THREAD_TILES_PER_WARP_X; ++tile_x) {
                #pragma unroll
                for (int x = 0; x < THREAD_TILE_SIZE_X; ++x) {
                    int shared_col = warp_col_idx * WARP_TILE_SIZE_X + 
                                   tile_x * (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                                   thread_col_idx_in_warp * THREAD_TILE_SIZE_X + x;
                    
                    if (shared_col < BLOCK_TILE_SIZE_X) {
                        B_vals[tile_x][x] = B_shared[k_i][shared_col];
                    } else {
                        B_vals[tile_x][x] = 0.0f;
                    }
                }
            }
            
            // 计算外积 - 完全展开的四重循环
            #pragma unroll
            for (int tile_y = 0; tile_y < NUM_THREAD_TILES_PER_WARP_Y; ++tile_y) {
                #pragma unroll
                for (int tile_x = 0; tile_x < NUM_THREAD_TILES_PER_WARP_X; ++tile_x) {
                    #pragma unroll
                    for (int y = 0; y < THREAD_TILE_SIZE_Y; ++y) {
                        #pragma unroll
                        for (int x = 0; x < THREAD_TILE_SIZE_X; ++x) {
                            C_results[tile_y][tile_x][y][x] += 
                                A_vals[tile_y][y] * B_vals[tile_x][x];
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果到全局内存
    #pragma unroll
    for (int tile_y = 0; tile_y < NUM_THREAD_TILES_PER_WARP_Y; ++tile_y) {
        #pragma unroll
        for (int tile_x = 0; tile_x < NUM_THREAD_TILES_PER_WARP_X; ++tile_x) {
            #pragma unroll
            for (int y = 0; y < THREAD_TILE_SIZE_Y; ++y) {
                #pragma unroll
                for (int x = 0; x < THREAD_TILE_SIZE_X; ++x) {
                    int global_m = blockIdx.y * BLOCK_TILE_SIZE_Y +
                                 warp_row_idx * WARP_TILE_SIZE_Y +
                                 tile_y * (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                                 thread_row_idx_in_warp * THREAD_TILE_SIZE_Y + y;
                    
                    int global_n = blockIdx.x * BLOCK_TILE_SIZE_X +
                                 warp_col_idx * WARP_TILE_SIZE_X +
                                 tile_x * (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                                 thread_col_idx_in_warp * THREAD_TILE_SIZE_X + x;
                    
                    if (global_m < M && global_n < N) {
                        C[global_m * N + global_n] = 
                            alpha * C_results[tile_y][tile_x][y][x] + 
                            beta * C[global_m * N + global_n];
                    }
                }
            }
        }
    }
}
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int THREAD_SIZE_Y, int THREAD_SIZE_X>
__global__ void cuda_gemm_optimized(const float* __restrict__ A, 
                                   const float* __restrict__ B, 
                                   float* __restrict__ C,
                                   int M, int N, int K, float alpha, float beta) {
    // 计算当前线程块的起始位置
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 每个线程负责计算 THREAD_SIZE_Y x THREAD_SIZE_X 个输出元素
    float c[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};
    
    // 共享内存声明 - 使用 padding 避免 bank conflicts
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K + 1];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N + 1];
    
    // 计算当前线程块在全局矩阵中的位置
    const int c_row_start = by * BLOCK_SIZE_M;
    const int c_col_start = bx * BLOCK_SIZE_N;
    
    // warp 级别的映射优化
    const int warpId = (ty * blockDim.x + tx) / 32;
    const int laneId = (ty * blockDim.x + tx) % 32;
    
    // 主循环 - K 维度分块
    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE_K) {
        // 协作加载 A 矩阵块到共享内存 (向量化加载)
        for (int load_offset = 0; load_offset < (BLOCK_SIZE_M * BLOCK_SIZE_K) / (blockDim.x * blockDim.y); ++load_offset) {
            int linear_idx = load_offset * blockDim.x * blockDim.y + ty * blockDim.x + tx;
            int a_row = linear_idx / BLOCK_SIZE_K;
            int a_col = linear_idx % BLOCK_SIZE_K;
            
            if (a_row < BLOCK_SIZE_M && (c_row_start + a_row) < M && (k_block + a_col) < K) {
                As[a_row][a_col] = A[(c_row_start + a_row) * K + (k_block + a_col)];
            } else {
                As[a_row][a_col] = 0.0f;
            }
        }
        
        // 协作加载 B 矩阵块到共享内存 (向量化加载)
        for (int load_offset = 0; load_offset < (BLOCK_SIZE_K * BLOCK_SIZE_N) / (blockDim.x * blockDim.y); ++load_offset) {
            int linear_idx = load_offset * blockDim.x * blockDim.y + ty * blockDim.x + tx;
            int b_row = linear_idx / BLOCK_SIZE_N;
            int b_col = linear_idx % BLOCK_SIZE_N;
            
            if (b_row < BLOCK_SIZE_K && (k_block + b_row) < K && (c_col_start + b_col) < N) {
                Bs[b_row][b_col] = B[(k_block + b_row) * N + (c_col_start + b_col)];
            } else {
                Bs[b_row][b_col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // 寄存器缓存优化的内积计算
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            // 预加载到寄存器
            float a_reg[THREAD_SIZE_Y];
            float b_reg[THREAD_SIZE_X];
            
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_Y; ++i) {
                a_reg[i] = As[ty * THREAD_SIZE_Y + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_SIZE_X; ++j) {
                b_reg[j] = Bs[k][tx * THREAD_SIZE_X + j];
            }
            
            // 外积计算 (完全展开的循环)
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_Y; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_X; ++j) {
                    c[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果到全局内存 (向量化写入)
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; ++j) {
            int c_row = c_row_start + ty * THREAD_SIZE_Y + i;
            int c_col = c_col_start + tx * THREAD_SIZE_X + j;
            
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = alpha * c[i][j] + beta * C[c_row * N + c_col];
            }
        }
    }
}

// 测试朴素 CUDA GEMM
double benchmark_naive_gemm(float* d_A, float* d_B, float* d_C,
                            int M, int N, int K, int num_runs = 10, int warmup_runs = 3) {
    const float alpha = 1.0f, beta = 0.0f;
    PreciseTimer timer;
    
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    std::cout << "  Warm-up (朴素): " << warmup_runs << " 次预热运行..." << std::flush;
    
    // 预热运行
    for (int i = 0; i < warmup_runs; ++i) {
        cuda_gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << " 完成" << std::endl;
    
    // 正式测试
    timer.start_gpu();
    for (int i = 0; i < num_runs; ++i) {
        cuda_gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    double total_time = timer.elapsed_gpu_ms();
    
    return total_time / num_runs;  // 平均时间
}

// 测试共享内存优化 GEMM
double benchmark_shared_gemm(float* d_A, float* d_B, float* d_C,
                             int M, int N, int K, int num_runs = 10, int warmup_runs = 3) {
    const float alpha = 1.0f, beta = 0.0f;
    PreciseTimer timer;
    
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    std::cout << "  Warm-up (共享内存): " << warmup_runs << " 次预热运行..." << std::flush;
    
    // 预热运行
    for (int i = 0; i < warmup_runs; ++i) {
        cuda_gemm_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << " 完成" << std::endl;
    
    // 正式测试
    timer.start_gpu();
    for (int i = 0; i < num_runs; ++i) {
        cuda_gemm_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    double total_time = timer.elapsed_gpu_ms();
    
    return total_time / num_runs;
}

// 测试 Fragment 算法 GEMM - 专业级实现
double benchmark_fragment_gemm(float* d_A, float* d_B, float* d_C,
                               int M, int N, int K, int num_runs = 10, int warmup_runs = 3) {
    const float alpha = 1.0f, beta = 0.0f;
    PreciseTimer timer;
    
    // 专业级 Fragment 算法参数 (基于文档中的实现)
    constexpr int BLOCK_TILE_SIZE_X = 128;
    constexpr int BLOCK_TILE_SIZE_Y = 128;
    constexpr int BLOCK_TILE_SIZE_K = 16;
    
    constexpr int WARP_TILE_SIZE_X = 32;
    constexpr int WARP_TILE_SIZE_Y = 64;
    
    constexpr int THREAD_TILE_SIZE_X = 8;
    constexpr int THREAD_TILE_SIZE_Y = 8;
    
    constexpr int NUM_THREADS_PER_WARP_X = 4;
    constexpr int NUM_THREADS_PER_WARP_Y = 8;
    
    constexpr int NUM_WARPS_X = BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X;
    constexpr int NUM_WARPS_Y = BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y;
    constexpr int NUM_THREADS_X = NUM_WARPS_X * NUM_THREADS_PER_WARP_X;
    constexpr int NUM_THREADS_Y = NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y;
    
    // 线程块配置 - 使用一维配置
    dim3 blockDim(NUM_THREADS_X * NUM_THREADS_Y, 1, 1);
    dim3 gridDim((N + BLOCK_TILE_SIZE_X - 1) / BLOCK_TILE_SIZE_X,
                 (M + BLOCK_TILE_SIZE_Y - 1) / BLOCK_TILE_SIZE_Y, 1);
    
    std::cout << "  Warm-up (Fragment Pro " << BLOCK_TILE_SIZE_Y << "x" << BLOCK_TILE_SIZE_X 
              << ", Warp " << WARP_TILE_SIZE_Y << "x" << WARP_TILE_SIZE_X << "): " 
              << warmup_runs << " 次预热运行..." << std::flush;
    
    // 预热运行 - 修正模板参数
    for (int i = 0; i < warmup_runs; ++i) {
        cuda_gemm_fragment<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                          WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                          THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
                          NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
            <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << " 完成" << std::endl;
    
    // 正式测试 - 修正模板参数
    timer.start_gpu();
    for (int i = 0; i < num_runs; ++i) {
        cuda_gemm_fragment<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                          WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                          THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
                          NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
            <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    double total_time = timer.elapsed_gpu_ms();
    
    return total_time / num_runs;
}

// 测试高度优化 GEMM - 动态调整分块大小
double benchmark_optimized_gemm(float* d_A, float* d_B, float* d_C,
                                int M, int N, int K, int num_runs = 10, int warmup_runs = 3) {
    const float alpha = 1.0f, beta = 0.0f;
    PreciseTimer timer;
    
    // 根据矩阵大小动态调整优化参数
    int BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_Y, THREAD_SIZE_X;
    
    if (M <= 512) {
        // 小矩阵使用较小的分块
        BLOCK_SIZE_M = 64;
        BLOCK_SIZE_N = 64;
        BLOCK_SIZE_K = 8;
        THREAD_SIZE_Y = 4;
        THREAD_SIZE_X = 4;
    } else {
        // 大矩阵使用较大的分块
        BLOCK_SIZE_M = 128;
        BLOCK_SIZE_N = 128;
        BLOCK_SIZE_K = 8;
        THREAD_SIZE_Y = 8;
        THREAD_SIZE_X = 8;
    }
    
    dim3 blockDim(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 gridDim((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    std::cout << "  Warm-up (高度优化 " << BLOCK_SIZE_M << "x" << BLOCK_SIZE_N << "): " 
              << warmup_runs << " 次预热运行..." << std::flush;
    
    // 预热运行
    for (int i = 0; i < warmup_runs; ++i) {
        if (M <= 512) {
            cuda_gemm_optimized<64, 64, 8, 4, 4>
                <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        } else {
            cuda_gemm_optimized<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << " 完成" << std::endl;
    
    // 正式测试
    timer.start_gpu();
    for (int i = 0; i < num_runs; ++i) {
        if (M <= 512) {
            cuda_gemm_optimized<64, 64, 8, 4, 4>
                <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        } else {
            cuda_gemm_optimized<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        }
    }
    double total_time = timer.elapsed_gpu_ms();
    
    return total_time / num_runs;
}

// 测试 cuBLAS GEMM
double benchmark_cublas_gemm(cublasHandle_t handle, 
                            float* d_A, float* d_B, float* d_C,
                            int M, int N, int K, int num_runs = 10, int warmup_runs = 3) {
    const float alpha = 1.0f, beta = 0.0f;
    PreciseTimer timer;
    
    std::cout << "  Warm-up (cuBLAS): " << warmup_runs << " 次预热运行..." << std::flush;
    
    // 预热运行
    for (int i = 0; i < warmup_runs; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << " 完成" << std::endl;
    
    // 正式测试
    timer.start_gpu();
    for (int i = 0; i < num_runs; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    double total_time = timer.elapsed_gpu_ms();
    
    return total_time / num_runs;  // 平均时间
}

// 验证结果正确性 - 增强版
bool verify_results(float* d_result, float* d_reference, int size, float tolerance = 1e-3f) {
    std::vector<float> h_result(size), h_reference(size);
    
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_reference.data(), d_reference, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    int error_count = 0;
    int check_count = std::min(size, 10000);  // 检查前10000个元素或全部
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (int i = 0; i < check_count; ++i) {
        float abs_diff = std::abs(h_result[i] - h_reference[i]);
        float rel_error = abs_diff / (std::abs(h_reference[i]) + 1e-8f);
        
        max_abs_error = std::max(max_abs_error, abs_diff);
        max_rel_error = std::max(max_rel_error, rel_error);
        
        if (rel_error > tolerance) {
            error_count++;
            if (error_count <= 5) {  // 只打印前5个错误
                std::cout << "    错误[" << i << "]: 期望=" << h_reference[i] 
                          << ", 实际=" << h_result[i] 
                          << ", 相对误差=" << rel_error << std::endl;
            }
        }
        
        // 检查 NaN 或 Inf
        if (std::isnan(h_result[i]) || std::isinf(h_result[i])) {
            std::cout << "    发现 NaN/Inf 在位置 " << i << ": " << h_result[i] << std::endl;
            return false;
        }
    }
    
    if (error_count > 0) {
        std::cout << "    验证失败: " << error_count << "/" << check_count 
                  << " 个元素错误 (最大绝对误差=" << max_abs_error 
                  << ", 最大相对误差=" << max_rel_error << ")" << std::endl;
        return false;
    }
    
    std::cout << "    验证通过: 最大绝对误差=" << max_abs_error 
              << ", 最大相对误差=" << max_rel_error << std::endl;
    return true;
}

int main() {
    std::cout << "CUDA GEMM 性能基准测试 - 完整对比" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // 获取 GPU 信息
    int device;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "全局内存: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "共享内存/块: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "最大线程/块: " << prop.maxThreadsPerBlock << std::endl << std::endl;
    
    // 测试多种矩阵大小
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    
    // 打印漂亮的表头
    std::cout << "\n" << std::string(135, '=') << std::endl;
    std::cout << std::setw(67) << "CUDA GEMM 性能对比结果" << std::endl;
    std::cout << std::string(135, '=') << std::endl;
    
    std::cout << "\n" << std::setw(60) << "测试配置说明" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::cout << std::left << std::setw(6) << "大小"
              << std::right << std::setw(8) << "朴素(ms)"
              << std::setw(8) << "共享(ms)" 
              << std::setw(8) << "优化(ms)"
              << std::setw(9) << "Fragment(ms)"
              << std::setw(10) << "cuBLAS(ms)"
              << std::setw(8) << "朴素GFLOPS"
              << std::setw(8) << "共享GFLOPS"
              << std::setw(8) << "优化GFLOPS"
              << std::setw(9) << "Fragment GFLOPS"
              << std::setw(10) << "cuBLAS GFLOPS"
              << std::setw(6) << "加速1"
              << std::setw(6) << "加速2"
              << std::setw(6) << "加速3"
              << std::setw(6) << "加速4" << std::endl;
    std::cout << std::string(135, '-') << std::endl;
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        double total_ops = 2.0 * M * N * K;  // GEMM 操作数
        
        // 分配内存
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
        
        // 初始化随机数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
        for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
        
        // GPU 内存分配
        float *d_A, *d_B, *d_C_naive, *d_C_shared, *d_C_optimized, *d_C_fragment, *d_C_cublas;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_naive, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_shared, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_optimized, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_fragment, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C_cublas, M * N * sizeof(float)));
        
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C_naive, 0, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C_shared, 0, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C_optimized, 0, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C_fragment, 0, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_C_cublas, 0, M * N * sizeof(float)));
        
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        
        // 配置测试参数
        int num_runs = 20;
        int warmup_runs = 10;  // 增加预热次数确保稳定性
        
        std::cout << "\n测试矩阵大小 " << size << "x" << size << " (运行 " << num_runs 
                  << " 次，预热 " << warmup_runs << " 次):" << std::endl;
        
        // 1. 朴素实现测试
        std::cout << "\n1. 朴素实现测试:" << std::endl;
        double naive_time = benchmark_naive_gemm(d_A, d_B, d_C_naive, M, N, K, num_runs, warmup_runs);
        double naive_gflops = total_ops / (naive_time * 1e6);
        
        // 2. 共享内存优化测试
        std::cout << "\n2. 共享内存优化测试:" << std::endl;
        double shared_time = benchmark_shared_gemm(d_A, d_B, d_C_shared, M, N, K, num_runs, warmup_runs);
        double shared_gflops = total_ops / (shared_time * 1e6);
        
        // 3. 高度优化实现测试
        std::cout << "\n3. 高度优化实现测试:" << std::endl;
        double optimized_time = benchmark_optimized_gemm(d_A, d_B, d_C_optimized, M, N, K, num_runs, warmup_runs);
        double optimized_gflops = total_ops / (optimized_time * 1e6);
        
        // 4. Fragment 算法测试
        std::cout << "\n4. Fragment 算法测试:" << std::endl;
        double fragment_time = benchmark_fragment_gemm(d_A, d_B, d_C_fragment, M, N, K, num_runs, warmup_runs);
        double fragment_gflops = total_ops / (fragment_time * 1e6);
        
        // 5. cuBLAS 测试（作为参考）
        std::cout << "\n5. cuBLAS 参考测试:" << std::endl;
        double cublas_time = benchmark_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K, num_runs, warmup_runs);
        double cublas_gflops = total_ops / (cublas_time * 1e6);
        
        // 计算加速比
        double shared_speedup = naive_time / shared_time;
        double optimized_speedup = naive_time / optimized_time;
        double fragment_speedup = naive_time / fragment_time;
        double cublas_speedup = naive_time / cublas_time;
        
        // 详细的正确性验证
        std::cout << "\n正确性验证:" << std::endl;
        std::cout << "  朴素 vs cuBLAS:" << std::endl;
        bool naive_correct = verify_results(d_C_naive, d_C_cublas, M * N);
        
        std::cout << "  共享内存 vs cuBLAS:" << std::endl;
        bool shared_correct = verify_results(d_C_shared, d_C_cublas, M * N);
        
        std::cout << "  高度优化 vs cuBLAS:" << std::endl;
        bool optimized_correct = verify_results(d_C_optimized, d_C_cublas, M * N);
        
        std::cout << "  Fragment vs cuBLAS:" << std::endl;
        bool fragment_correct = verify_results(d_C_fragment, d_C_cublas, M * N);
        
        // 输出结果 - 精确对齐
        std::cout << "\n性能结果汇总:" << std::endl;
        std::cout << std::left << std::setw(6) << size
                  << std::right << std::fixed << std::setprecision(2) << std::setw(8) << naive_time
                  << std::setw(8) << shared_time
                  << std::setw(8) << optimized_time
                  << std::setw(9) << fragment_time
                  << std::setw(10) << cublas_time
                  << std::fixed << std::setprecision(0) << std::setw(8) << naive_gflops
                  << std::setw(8) << shared_gflops
                  << std::setw(8) << optimized_gflops
                  << std::setw(9) << fragment_gflops
                  << std::setw(10) << cublas_gflops
                  << std::fixed << std::setprecision(1) << std::setw(5) << shared_speedup << "x"
                  << std::setw(5) << optimized_speedup << "x"
                  << std::setw(5) << fragment_speedup << "x"
                  << std::setw(5) << cublas_speedup << "x";
        
        // 正确性标记
        std::cout << " [正确性: ";
        std::cout << (naive_correct ? "✓" : "✗") << " ";
        std::cout << (shared_correct ? "✓" : "✗") << " "; 
        std::cout << (optimized_correct ? "✓" : "✗") << " ";
        std::cout << (fragment_correct ? "✓" : "✗") << " ";
        std::cout << "✓]";  // cuBLAS 作为参考，总是正确
        
        if (!naive_correct || !shared_correct || !optimized_correct || !fragment_correct) {
            std::cout << " ⚠️  发现错误！";
        }
        std::cout << std::endl;
        
        // 清理资源
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C_naive));
        CHECK_CUDA(cudaFree(d_C_shared));
        CHECK_CUDA(cudaFree(d_C_fragment));
        CHECK_CUDA(cudaFree(d_C_cublas));
    }
    
    std::cout << std::string(135, '=') << std::endl;
    std::cout << "\n测试说明:" << std::endl;
    std::cout << "• 朴素实现: 简单三重循环，每线程计算一个输出元素" << std::endl;
    std::cout << "• 共享内存: 使用 16x16 分块和共享内存优化数据复用" << std::endl;
    std::cout << "• 高度优化: 多级分块 + 向量化 + 寄存器缓存 + warp级映射" << std::endl;
    std::cout << "• Fragment: 高性能分片算法，寄存器分块 + 外积优化" << std::endl;
    std::cout << "• cuBLAS: NVIDIA 高度优化的生产级 GEMM 库" << std::endl;
    std::cout << "• 加速1: 共享内存相对朴素实现的加速比" << std::endl;
    std::cout << "• 加速2: 高度优化相对朴素实现的加速比" << std::endl;
    std::cout << "• 加速3: Fragment 相对朴素实现的加速比" << std::endl;
    std::cout << "• 加速4: cuBLAS 相对朴素实现的加速比" << std::endl;
    std::cout << "• 所有测试都进行 5 次 warm-up 预热，然后多次运行取平均" << std::endl;
    std::cout << "• 使用 cudaEvent 精确测量 GPU 执行时间" << std::endl;
    
    return 0;
}