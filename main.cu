#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <random>

using namespace std::chrono;

__global__
void init_curand(curandState* state, int numElements)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < numElements)
	{
		curand_init(id, 0, 0, &state[id]);
	}
}

__global__
void randomize(curandState* state, float* a, int numElements)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numElements)
	{
		a[id] = curand_uniform(state + id);
	}
}

__global__
void add(float* a, float* b, float* out, int numElements)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numElements)
	{
		out[id] = a[id] + b[id];
	}
}

auto runCudaTestA(int num_elements)
{
	auto test_begin = high_resolution_clock::now();

	float* x, * y, * z;
	cudaMalloc(&x, num_elements * sizeof(float));
	cudaMalloc(&y, num_elements * sizeof(float));
	cudaMalloc(&z, num_elements * sizeof(float));

	constexpr int blockSize = 256;
	int numBlocks = ((num_elements - 1) / blockSize) + 1;
	curandState* randomStates;
	cudaMalloc(&randomStates, sizeof(curandState) * num_elements * 2);

	init_curand<<<numBlocks, blockSize>>> (randomStates, num_elements);
	cudaDeviceSynchronize();
	randomize<<<numBlocks, blockSize>>> (randomStates, x, num_elements);
	randomize<<<numBlocks, blockSize>>>(randomStates + num_elements, y, num_elements);
	cudaDeviceSynchronize();
	add<<<numBlocks, blockSize >> > (x, y, z, num_elements);
	cudaDeviceSynchronize();
	cudaFree(randomStates);
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);

	auto test_end = high_resolution_clock::now();

	return test_end - test_begin;
}

struct collection
{
	float x, y, z;
};

__global__
void randomize_b(collection* collections, int num_elements)
{
	curandState state;
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < num_elements)
	{
		curand_init(id, 0, 0, &state);
		collections[id].x = curand_uniform(&state);
		collections[id].y = curand_uniform(&state);
	}
}

__global__
void add_b(collection* collections, int num_elements)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < num_elements)
	{
		collections[id].z = collections[id].x + collections[id].y;
	}
}

auto runCudaTestB(int num_elements)
{
	auto test_begin = high_resolution_clock::now();

	collection* collections;
	cudaMalloc(&collections, num_elements * sizeof(collection));

	constexpr int blockSize = 256;
	int numBlocks = ((num_elements - 1) / blockSize) + 1;
	randomize_b<<<numBlocks, blockSize>>>(collections, num_elements);
	cudaDeviceSynchronize();
	add_b<<<numBlocks, blockSize>>>(collections, num_elements);
	cudaDeviceSynchronize();

	cudaFree(collections);

	auto test_end = high_resolution_clock::now();

	return test_end - test_begin;
}

void randomize_cpu(std::mt19937& generator, std::uniform_real_distribution<float>& distribution, float* a, float num_elements)
{
	for (int i = 0; i < num_elements; ++i)
	{
		a[i] = distribution(generator);
	}
}

void add_cpu(float* a, float* b, float* out, float num_elements)
{
	for (int i = 0; i < num_elements; ++i)
	{
		out[i] = a[i] + b[i];
	}
}

auto runCPUTest(int num_elements)
{
	auto test_begin = high_resolution_clock::now();

	float* x = new float[num_elements];
	float* y = new float[num_elements];
	float* z = new float[num_elements];

	std::mt19937 randomGenerator(0);
	std::uniform_real_distribution<float> distribution;

	randomize_cpu(randomGenerator, distribution, x, num_elements);
	randomize_cpu(randomGenerator, distribution, y, num_elements);
	randomize_cpu(randomGenerator, distribution, z, num_elements);

	add_cpu(x, y, z, num_elements);

	delete[] x;
	delete[] y;
	delete[] z;

	auto test_end = high_resolution_clock::now();

	return test_end - test_begin;
}

int main()
{
	constexpr int num_elements = 1 << 25;

	std::cout << "Num elements: " << num_elements << std::endl;
	std::cout << "CUDA test A: " << duration<float, milliseconds::period>(runCudaTestA(num_elements)).count() << std::endl;
	std::cout << "CUDA test B: " << duration<float, milliseconds::period>(runCudaTestB(num_elements)).count() << std::endl;
	std::cout << "CPU test: " << duration<float, milliseconds::period>(runCPUTest(num_elements)).count() << std::endl;
	// Will add SIMD test later...
}