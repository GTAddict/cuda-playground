#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <random>

using namespace std::chrono;

__global__
void init_curand(curandState* state, unsigned long seed, int num_elements)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < num_elements)
	{
		curand_init(seed, id, 0, &state[id]);
	}
}

__global__
void randomize(curandState* state, float* a, int num_elements)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < num_elements)
	{
		a[id] = curand_uniform(state + id);
	}
}

__global__
void add(float* a, float* b, float* out, int num_elements)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < num_elements)
	{
		out[id] = a[id] + b[id];
	}
}

auto runCudaTest(float num_elements)
{
	auto test_begin = high_resolution_clock::now();

	float* x, * y, * z;
	cudaMallocManaged(&x, num_elements * sizeof(float));
	cudaMallocManaged(&y, num_elements * sizeof(float));
	cudaMallocManaged(&z, num_elements * sizeof(float));

	constexpr int blockSize = 256;
	int numBlocks = ((num_elements - 1) / blockSize) + 1;
	curandState* randomStates;
	cudaMalloc(&randomStates, sizeof(curandState) * num_elements);

	init_curand<<<numBlocks, blockSize >> > (randomStates, 0, num_elements);
	cudaDeviceSynchronize();

	randomize<<<numBlocks, blockSize>>> (randomStates, x, num_elements);
	randomize<<<numBlocks, blockSize>>>(randomStates, y, num_elements);
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
	constexpr int num_elements = 1 << 29;

	std::cout << "Num elements: " << num_elements << std::endl;
	std::cout << "CUDA test: " << duration<float, milliseconds::period>(runCudaTest(num_elements)).count() << std::endl;
	std::cout << "CPU test: " << duration<float, milliseconds::period>(runCPUTest(num_elements)).count() << std::endl;
	// Will add SIMD test later...
}