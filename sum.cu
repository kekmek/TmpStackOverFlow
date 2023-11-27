#include <iostream>
#include <random>
#include <chrono>

#include <cuda_runtime.h>

const int num_seeds = 20;

__global__ void vectorAdd(float* a, float* b, float* c, int vector_size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < vector_size) {
        c[index] = a[index] + b[index];
    }
}

void printMatrix(float* a, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {

    int N = std::stoi(argv[1]);

    for (int j = 0; j < num_seeds; ++j) {

        int size = N * sizeof(float);

        float* host_a = nullptr;
        float* host_b = nullptr;
        float* host_c = nullptr;

        // cudaMallocHost((void**)& host_a, size);
        // cudaMallocHost((void**)& host_b, size);   
        // cudaMallocHost((void**)& host_c, size);

        host_a = new float[N]{};
        host_b = new float[N]{};
        host_c = new float[N]{};

        float* device_a, *device_b, *device_c;

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist;

        for (int i = 0; i < N; ++i) {
            host_a[i] = dist(mt);
            host_b[i] = dist(mt);
        }

        dim3 blockDim(1024, 1, 1);
        dim3 dimGrid((N + blockDim.x - 1) / blockDim.x, 1, 1);

        cudaMalloc((void**)&device_a, size);
        cudaMalloc((void**)&device_b, size);
        cudaMalloc((void**)&device_c, size);

        cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

        vectorAdd<<<dimGrid, blockDim>>>(device_a, device_b, device_c, N);
        
        cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);
        
        // printMatrix(host_a, N);
        // printMatrix(host_b, N);
        // printMatrix(host_c, N);

        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);

        // cudaFreeHost(host_a);
        // cudaFreeHost(host_b);
        // cudaFreeHost(host_c);

        delete[] host_a;
        delete[] host_b;
        delete[] host_c;
    }
    return 0;
}