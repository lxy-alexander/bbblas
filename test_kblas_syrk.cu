#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d status=%d\n",__FILE__,__LINE__,(int)s); exit(2);} } while(0)

#define tx threadIdx.x
#define ty threadIdx.y

template<typename T> __device__ __forceinline__ T make_zero(){return T(0);}
__device__ __forceinline__ double FMA(double a,double b,double c){return fma(a,b,c);}
template<typename T> __device__ __forceinline__ T shfl(T v,int src,int width){
  return __shfl_sync(0xffffffffu,v,src,width);
}

// ==================== 你提供的 kernel 原样 ====================
template<typename T, int B_ROWS, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_Mfix_Nmul( const int m, const int n,
                                   const T alpha, const T* __restrict__ A, int lda,
                                   const T beta, T* B, int ldb)
{
  T rA0[A_COLS_PTY], s, rB0[B_ROWS], zero = make_zero<T>();
  int blockCount = n / A_COLS_PTY, ind;
  #pragma unroll
  for(int i = 0; i < B_ROWS; i++){
    rB0[ i ] = __ldg(&(B[ tx + i * ldb ])) * beta ;
  }
  for(int b = 0; b < blockCount; b++){
    ind = tx  + A_COLS_PTY * b * lda;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
    }
    #pragma unroll
    for(int j = 0; j < B_ROWS; j++){
      s = zero;
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        s = FMA(rA0[i], shfl(rA0[i], j, B_ROWS), s);
      }
      rB0[j] = FMA( alpha, s, rB0[j] );
    }
  }
  #pragma unroll
  for(int i = 0; i < B_ROWS; i++)
  {
    if(tx >= i)
      B[ tx + i * ldb ] = rB0[ i ];
  }
}

template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void
kernel_syrk_U_LN_registers_Mfix_Nmul( const int m, const int n, int batchCount,
                                      const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                      const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m != B_ROWS ) return;
  if( n % A_COLS_PTY ) return;
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  if(ind >= batchCount) return;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_Mfix_Nmul<T, B_ROWS, A_COLS_PTY>(
                                    m, n,
                                    alpha, A, lda,
                                    beta, B, ldb);
}
// =======================================================

// allclose 检查
struct CompareStats {
  bool ok; double maxAbs, maxRel;
};
CompareStats compareResults(const double* ref,const double* out,int n,int ld,double rtol=1e-5,double atol=1e-8){
  double maxA=0,maxR=0; bool ok=true;
  for(int j=0;j<n;++j)for(int i=0;i<n;++i){
    double a=ref[i+j*ld], b=out[i+j*ld];
    double abs=std::abs(a-b);
    double rel=abs/(std::abs(a)+1e-12);
    if(!(abs<=atol+rtol*std::abs(a))) ok=false;
    maxA=std::max(maxA,abs); maxR=std::max(maxR,rel);
  }
  return {ok,maxA,maxR};
}

int main(){
  const int N=64, K=64, batchCount=512;
  const int lda=N, ldc=N;
  const double alpha=1.0, beta=0.0;

  // host 初始化
  std::vector<double> hA(size_t(N)*K*batchCount), hC1(size_t(N)*N*batchCount), hC2(size_t(N)*N*batchCount);
  for(size_t b=0;b<batchCount;++b){
    for(int j=0;j<K;++j)
      for(int i=0;i<N;++i)
        hA[b*N*K + i+j*lda] = std::sin(0.01*(i+1)*(j+1+b));
    std::fill(hC1.begin()+b*N*N, hC1.begin()+(b+1)*N*N, 0.0);
    std::fill(hC2.begin()+b*N*N, hC2.begin()+(b+1)*N*N, 0.0);
  }

  // device 分配
  double *dA,*dC1,*dC2;
  CHECK_CUDA(cudaMalloc(&dA, sizeof(double)*hA.size()));
  CHECK_CUDA(cudaMalloc(&dC1,sizeof(double)*hC1.size()));
  CHECK_CUDA(cudaMalloc(&dC2,sizeof(double)*hC2.size()));
  CHECK_CUDA(cudaMemcpy(dA,hA.data(),sizeof(double)*hA.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC1,hC1.data(),sizeof(double)*hC1.size(),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC2,hC2.data(),sizeof(double)*hC2.size(),cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // ========= cuBLAS 测试 (for 循环调用 Dsyrk) =========
  cudaEvent_t s1,e1; CHECK_CUDA(cudaEventCreate(&s1)); CHECK_CUDA(cudaEventCreate(&e1));
  for(int w=0; w<5; ++w){
    for (int b=0; b<batchCount; ++b){
      CHECK_CUBLAS(cublasDsyrk(handle,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        N, K,
        &alpha,
        dA + b*N*K, lda,
        &beta,
        dC1 + b*N*N, ldc));
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(s1));
  for(int it=0; it<20; ++it){
    for (int b=0; b<batchCount; ++b){
      CHECK_CUBLAS(cublasDsyrk(handle,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        N, K,
        &alpha,
        dA + b*N*K, lda,
        &beta,
        dC1 + b*N*N, ldc));
    }
  }
  CHECK_CUDA(cudaEventRecord(e1));
  CHECK_CUDA(cudaEventSynchronize(e1));
  float ms1; CHECK_CUDA(cudaEventElapsedTime(&ms1,s1,e1)); ms1/=20.0f;

  // ========= 自定义 kernel =========
  dim3 block(64,1,1), grid(batchCount,1,1);
  cudaEvent_t s2,e2; CHECK_CUDA(cudaEventCreate(&s2)); CHECK_CUDA(cudaEventCreate(&e2));
  for(int w=0; w<5; ++w){
    kernel_syrk_U_LN_registers_Mfix_Nmul<double,double*,true,64,8>
      <<<grid,block>>>(64,64,batchCount,alpha,dA,0,0,lda,N*K,beta,dC2,0,0,ldc,N*N);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(s2));
  for(int it=0; it<20; ++it){
    kernel_syrk_U_LN_registers_Mfix_Nmul<double,double*,true,64,8>
      <<<grid,block>>>(64,64,batchCount,alpha,dA,0,0,lda,N*K,beta,dC2,0,0,ldc,N*N);
  }
  CHECK_CUDA(cudaEventRecord(e2));
  CHECK_CUDA(cudaEventSynchronize(e2));
  float ms2; CHECK_CUDA(cudaEventElapsedTime(&ms2,s2,e2)); ms2/=20.0f;

  // ========= 校验 =========
  CHECK_CUDA(cudaMemcpy(hC1.data(),dC1,sizeof(double)*hC1.size(),cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hC2.data(),dC2,sizeof(double)*hC2.size(),cudaMemcpyDeviceToHost));
  CompareStats st = compareResults(hC1.data(),hC2.data(),N,ldc);

  printf("=== Performance Results ===\n");
  printf("Matrix size: %dx%d batch=%d\n",N,N,batchCount);
  printf("cuBLAS Dsyrk: %.6f ms (looped)\n", ms1);
  printf("Custom kernel: %.6f ms\n", ms2);
  printf("Speedup: %.4fx\n", ms1/ms2);
  printf("Max abs err: %.3e  Max rel err: %.3e\n", st.maxAbs, st.maxRel);
  if(st.ok) printf("✓ Results match\n"); else printf("✗ Mismatch\n");

  return 0;
}
