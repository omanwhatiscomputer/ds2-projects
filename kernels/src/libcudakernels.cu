#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// op codes: 0=add, 1=sub, 2=mul, 3=div
__global__ void vectorOpKernel(const double *a, const double *b, double *c, int n, int op) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        switch (op) {
            case 0: c[i] = a[i] + b[i]; break;
            case 1: c[i] = a[i] - b[i]; break;
            case 2: c[i] = a[i] * b[i]; break;
            case 3: c[i] = a[i] / b[i]; break;
        }
    }
}

// op codes: 0=add, 1=sub, 2=mul, 3=div  (scalar broadcast)
__global__ void vectorScalarOpKernel(const double *a, double scalar, double *c, int n, int op) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        switch (op) {
            case 0: c[i] = a[i] + scalar; break;
            case 1: c[i] = a[i] - scalar; break;
            case 2: c[i] = a[i] * scalar; break;
            case 3: c[i] = a[i] / scalar; break;
        }
    }
}

extern "C" void gpuVectorScalarOp(const double* h_a, double scalar, double* h_result, int n, int op) {
    double *d_a, *d_c;
    size_t size = n * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorScalarOpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, scalar, d_c, n, op);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_c);
}

extern "C" void gpuVectorOp(const double* h_a, const double* h_b, double* h_result, int n, int op) {
    double *d_a, *d_b, *d_c;
    size_t size = n * sizeof(double);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorOpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n, op);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Matrix wrappers — flatten rows*cols to a 1D buffer and delegate to vector kernels
extern "C" void gpuMatrixOp(const double* h_a, const double* h_b, double* h_result, int rows, int cols, int op) {
    gpuVectorOp(h_a, h_b, h_result, rows * cols, op);
}

extern "C" void gpuMatrixScalarOp(const double* h_a, double scalar, double* h_result, int rows, int cols, int op) {
    gpuVectorScalarOp(h_a, scalar, h_result, rows * cols, op);
}

// Row broadcast: c[k] = op(a[k], vec[k % cols])
__global__ void matrixRowVecOpKernel(const double *a, const double *vec, double *c, int n, int cols, int op) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) {
        int j = k % cols;
        switch (op) {
            case 0: c[k] = a[k] + vec[j]; break;
            case 1: c[k] = a[k] - vec[j]; break;
            case 2: c[k] = a[k] * vec[j]; break;
            case 3: c[k] = a[k] / vec[j]; break;
        }
    }
}

extern "C" void gpuMatrixRowVecOp(const double* h_a, const double* h_vec, double* h_result, int rows, int cols, int op) {
    double *d_a, *d_vec, *d_c;
    int n = rows * cols;
    size_t matSize = n * sizeof(double);
    size_t vecSize = cols * sizeof(double);

    cudaMalloc((void**)&d_a,   matSize);
    cudaMalloc((void**)&d_vec, vecSize);
    cudaMalloc((void**)&d_c,   matSize);

    cudaMemcpy(d_a,   h_a,   matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, vecSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    matrixRowVecOpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_vec, d_c, n, cols, op);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_c, matSize, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_vec); cudaFree(d_c);
}

// Col broadcast: c[k] = op(a[k], vec[k / cols])
__global__ void matrixColVecOpKernel(const double *a, const double *vec, double *c, int n, int cols, int op) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) {
        int i = k / cols;
        switch (op) {
            case 0: c[k] = a[k] + vec[i]; break;
            case 1: c[k] = a[k] - vec[i]; break;
            case 2: c[k] = a[k] * vec[i]; break;
            case 3: c[k] = a[k] / vec[i]; break;
        }
    }
}

extern "C" void gpuMatrixColVecOp(const double* h_a, const double* h_vec, double* h_result, int rows, int cols, int op) {
    double *d_a, *d_vec, *d_c;
    int n = rows * cols;
    size_t matSize = n * sizeof(double);
    size_t vecSize = rows * sizeof(double);

    cudaMalloc((void**)&d_a,   matSize);
    cudaMalloc((void**)&d_vec, vecSize);
    cudaMalloc((void**)&d_c,   matSize);

    cudaMemcpy(d_a,   h_a,   matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, vecSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    matrixColVecOpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_vec, d_c, n, cols, op);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_c, matSize, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_vec); cudaFree(d_c);
}

// %% cuBLAS handle (lazy-initialized, shared across calls) %%%%%%%%%%%%%%%%%

static cublasHandle_t g_cublasHandle = nullptr;

static cublasHandle_t getCublasHandle() {
    if (g_cublasHandle == nullptr) {
        cublasCreate(&g_cublasHandle);
    }
    return g_cublasHandle;
}

// %% GEMM: C = A * B  (all row-major, A is m×k, B is k×n, C is m×n) %%%%%%%
// cuBLAS is column-major; we compute C^T = B^T * A^T to stay row-major.
extern "C" void gpuMatrixMul(const double* h_a, const double* h_b, double* h_c, int m, int k, int n) {
    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, (size_t)m * k * sizeof(double));
    cudaMalloc(&d_b, (size_t)k * n * sizeof(double));
    cudaMalloc(&d_c, (size_t)m * n * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (size_t)k * n * sizeof(double), cudaMemcpyHostToDevice);
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(getCublasHandle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_b, n,
        d_a, k,
        &beta,
        d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, (size_t)m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// %% GEMV: result = A * v  (row-major A is m×n, v is n, result is m) %%%%%%%
extern "C" void gpuMatrixVecMul(const double* h_a, const double* h_v, double* h_result, int m, int n) {
    double *d_a, *d_v, *d_r;
    cudaMalloc(&d_a, (size_t)m * n * sizeof(double));
    cudaMalloc(&d_v, (size_t)n     * sizeof(double));
    cudaMalloc(&d_r, (size_t)m     * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, (size_t)n     * sizeof(double), cudaMemcpyHostToDevice);
    const double alpha = 1.0, beta = 0.0;
    // row-major A treated as col-major A^T; CUBLAS_OP_T un-transposes it
    cublasDgemv(getCublasHandle(), CUBLAS_OP_T, n, m, &alpha, d_a, n, d_v, 1, &beta, d_r, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_r, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_v); cudaFree(d_r);
}

// %% Transposed GEMV: result = A^T * v  (row-major A is m×n, v is m, result is n) %
extern "C" void gpuMatrixTransVecMul(const double* h_a, const double* h_v, double* h_result, int m, int n) {
    double *d_a, *d_v, *d_r;
    cudaMalloc(&d_a, (size_t)m * n * sizeof(double));
    cudaMalloc(&d_v, (size_t)m     * sizeof(double));
    cudaMalloc(&d_r, (size_t)n     * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, (size_t)m     * sizeof(double), cudaMemcpyHostToDevice);
    const double alpha = 1.0, beta = 0.0;
    // row-major A^T = col-major A; CUBLAS_OP_N leaves it as-is
    cublasDgemv(getCublasHandle(), CUBLAS_OP_N, n, m, &alpha, d_a, n, d_v, 1, &beta, d_r, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_r, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_v); cudaFree(d_r);
}

// %% Column sums via GEMV with ones: sumV[j] = A^T * ones_m %%%%%%%%%%%%%%%
extern "C" void gpuMatrixColSum(const double* h_a, double* h_result, int rows, int cols) {
    double *d_a, *d_ones, *d_r;
    cudaMalloc(&d_a,    (size_t)rows * cols * sizeof(double));
    cudaMalloc(&d_ones, (size_t)rows        * sizeof(double));
    cudaMalloc(&d_r,    (size_t)cols        * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    double *h_ones = new double[rows];
    for (int i = 0; i < rows; i++) h_ones[i] = 1.0;
    cudaMemcpy(d_ones, h_ones, (size_t)rows * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_ones;
    const double alpha = 1.0, beta = 0.0;
    cublasDgemv(getCublasHandle(), CUBLAS_OP_N, cols, rows, &alpha, d_a, cols, d_ones, 1, &beta, d_r, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_r, (size_t)cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_ones); cudaFree(d_r);
}

// %% Row sums via GEMV with ones: sumVr[i] = A * ones_n %%%%%%%%%%%%%%%%%%%
extern "C" void gpuMatrixRowSum(const double* h_a, double* h_result, int rows, int cols) {
    double *d_a, *d_ones, *d_r;
    cudaMalloc(&d_a,    (size_t)rows * cols * sizeof(double));
    cudaMalloc(&d_ones, (size_t)cols        * sizeof(double));
    cudaMalloc(&d_r,    (size_t)rows        * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    double *h_ones = new double[cols];
    for (int i = 0; i < cols; i++) h_ones[i] = 1.0;
    cudaMemcpy(d_ones, h_ones, (size_t)cols * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_ones;
    const double alpha = 1.0, beta = 0.0;
    cublasDgemv(getCublasHandle(), CUBLAS_OP_T, cols, rows, &alpha, d_a, cols, d_ones, 1, &beta, d_r, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_r, (size_t)rows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_ones); cudaFree(d_r);
}

// %% Global sum reduction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void globalSumKernel(const double *a, double *partial, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? a[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

extern "C" void gpuMatrixGlobalSum(const double* h_a, double* h_result, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    double *d_a, *d_partial;
    cudaMalloc(&d_a,       (size_t)n      * sizeof(double));
    cudaMalloc(&d_partial, (size_t)blocks * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    globalSumKernel<<<blocks, threads, (size_t)threads * sizeof(double)>>>(d_a, d_partial, n);
    cudaDeviceSynchronize();
    double *h_partial = new double[blocks];
    cudaMemcpy(h_partial, d_partial, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < blocks; i++) sum += h_partial[i];
    *h_result = sum;
    delete[] h_partial;
    cudaFree(d_a); cudaFree(d_partial);
}

// %% Global min reduction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void globalMinKernel(const double *a, double *partial, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? a[i] : 1e300;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = min(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

extern "C" void gpuMatrixGlobalMin(const double* h_a, double* h_result, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    double *d_a, *d_partial;
    cudaMalloc(&d_a,       (size_t)n      * sizeof(double));
    cudaMalloc(&d_partial, (size_t)blocks * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    globalMinKernel<<<blocks, threads, (size_t)threads * sizeof(double)>>>(d_a, d_partial, n);
    cudaDeviceSynchronize();
    double *h_partial = new double[blocks];
    cudaMemcpy(h_partial, d_partial, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double result = h_partial[0];
    for (int i = 1; i < blocks; i++) if (h_partial[i] < result) result = h_partial[i];
    *h_result = result;
    delete[] h_partial;
    cudaFree(d_a); cudaFree(d_partial);
}

// %% Global max reduction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void globalMaxKernel(const double *a, double *partial, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? a[i] : -1e300;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

extern "C" void gpuMatrixGlobalMax(const double* h_a, double* h_result, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    double *d_a, *d_partial;
    cudaMalloc(&d_a,       (size_t)n      * sizeof(double));
    cudaMalloc(&d_partial, (size_t)blocks * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    globalMaxKernel<<<blocks, threads, (size_t)threads * sizeof(double)>>>(d_a, d_partial, n);
    cudaDeviceSynchronize();
    double *h_partial = new double[blocks];
    cudaMemcpy(h_partial, d_partial, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double result = h_partial[0];
    for (int i = 1; i < blocks; i++) if (h_partial[i] > result) result = h_partial[i];
    *h_result = result;
    delete[] h_partial;
    cudaFree(d_a); cudaFree(d_partial);
}

// %% Column min %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void colMinKernel(const double *A, double *result, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        double val = A[col];
        for (int i = 1; i < rows; i++) { double x = A[i * cols + col]; if (x < val) val = x; }
        result[col] = val;
    }
}

extern "C" void gpuMatrixColMin(const double* h_a, double* h_result, int rows, int cols) {
    double *d_a, *d_r;
    cudaMalloc(&d_a, (size_t)rows * cols * sizeof(double));
    cudaMalloc(&d_r, (size_t)cols        * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks  = (cols + threads - 1) / threads;
    colMinKernel<<<blocks, threads>>>(d_a, d_r, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_r, (size_t)cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_r);
}

// %% Column max %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void colMaxKernel(const double *A, double *result, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        double val = A[col];
        for (int i = 1; i < rows; i++) { double x = A[i * cols + col]; if (x > val) val = x; }
        result[col] = val;
    }
}

extern "C" void gpuMatrixColMax(const double* h_a, double* h_result, int rows, int cols) {
    double *d_a, *d_r;
    cudaMalloc(&d_a, (size_t)rows * cols * sizeof(double));
    cudaMalloc(&d_r, (size_t)cols        * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks  = (cols + threads - 1) / threads;
    colMaxKernel<<<blocks, threads>>>(d_a, d_r, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_r, (size_t)cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_r);
}

// %% Vector reductions (delegate to existing global reduction kernels) %%%%%

extern "C" void gpuVectorSum(const double* h_a, double* h_result, int n) {
    gpuMatrixGlobalSum(h_a, h_result, n);
}

extern "C" void gpuVectorMin(const double* h_a, double* h_result, int n) {
    gpuMatrixGlobalMin(h_a, h_result, n);
}

extern "C" void gpuVectorMax(const double* h_a, double* h_result, int n) {
    gpuMatrixGlobalMax(h_a, h_result, n);
}

// %% Vector dot product via cuBLAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

extern "C" void gpuVectorDot(const double* h_a, const double* h_b, double* h_result, int n) {
    double *d_a, *d_b;
    cudaMalloc(&d_a, (size_t)n * sizeof(double));
    cudaMalloc(&d_b, (size_t)n * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    double result;
    cublasDdot(getCublasHandle(), n, d_a, 1, d_b, 1, &result);
    cudaDeviceSynchronize();
    *h_result = result;
    cudaFree(d_a); cudaFree(d_b);
}

// %% normSq = dot(a, a) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

extern "C" void gpuVectorNormSq(const double* h_a, double* h_result, int n) {
    gpuVectorDot(h_a, h_a, h_result, n);
}

// %% norm = cublasDnrm2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

extern "C" void gpuVectorNorm(const double* h_a, double* h_result, int n) {
    double *d_a;
    cudaMalloc(&d_a, (size_t)n * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    double result;
    cublasDnrm2(getCublasHandle(), n, d_a, 1, &result);
    cudaDeviceSynchronize();
    *h_result = result;
    cudaFree(d_a);
}

// %% norm1 = sum of absolute values via cublasDasum %%%%%%%%%%%%%%%%%%%%%%%%

extern "C" void gpuVectorNorm1(const double* h_a, double* h_result, int n) {
    double *d_a;
    cudaMalloc(&d_a, (size_t)n * sizeof(double));
    cudaMemcpy(d_a, h_a, (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
    double result;
    cublasDasum(getCublasHandle(), n, d_a, 1, &result);
    cudaDeviceSynchronize();
    *h_result = result;
    cudaFree(d_a);
}
