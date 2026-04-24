#include <cuda_runtime.h>
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
