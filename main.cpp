#include <iostream>
#include <random>

void randomize(float* a)
{
	static std::mt19937 random_generator(0);
	*a = random_generator();
}

void add(int n, float* a, float* b, float* out)
{
	for (int i = 0; i < n; ++i)
	{
		out[i] = a[i] + b[i];
	}
}

int main()
{
	constexpr int num_elements = 1 << 20;

	float* x = new float[num_elements];
	float* y = new float[num_elements];
	float* z = new float[num_elements];

	for (int i = 0; i < num_elements; ++i)
	{
		randomize(x + i);
	}
	for (int i = 0; i < num_elements; ++i)
	{
		randomize(y + i);
	}

	for (int i = 0; i < num_elements; ++i)
	{
		add(num_elements, x, y, z);
	}

	delete[] x;
	delete[] y;
	delete[] z;
}