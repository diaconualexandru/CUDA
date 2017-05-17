#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>



#define MINREAL -1024.0
#define MAXREAL 1024.0
#define FAST_RED
#define ACCURACY 0.0001

#define NUM_OF_GPU_THREADS 256

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void vecFillRand(int N, float *vec) {
	int i;
	for (i = 0; i < N; i++)
		vec[i] = (rand() / (float)RAND_MAX)*(MAXREAL - MINREAL) + MINREAL;
}

float seq_dotProduct(float *a, float *b, int n) {
	int i;
	float dp;
	dp = 0;
	for (i = 0; i < n; i++) {
		dp += a[i] * b[i];
	}
	return dp;
}

// krenel
__global__ void dotProduct(float *a, float *b, float *c, int n) {
	__shared__ float temp[NUM_OF_GPU_THREADS];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
		temp[threadIdx.x] = a[idx] * b[idx];
	else temp[threadIdx.x] = 0.0f;

	__syncthreads();
#ifdef FAST_RED
	// assumes block dimension is a power of 2
	for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
		if (threadIdx.x < i) temp[threadIdx.x] += temp[threadIdx.x + i];
		__syncthreads();
	}
	if (threadIdx.x == 0) c[blockIdx.x] = temp[0];
#else
	float t;
	if (threadIdx.x == 0) {
		c[blockIdx.x] = 0.0f;
		int j = 0;
		for (int i = blockIdx.x*blockDim.x; ((i < ((blockIdx.x + 1)*blockDim.x)) && (i < n)); i++) {
			t = temp[j++];
			c[blockIdx.x] = c[blockIdx.x] + t;
		}
	}
#endif
}

int main(int argc, char* argv[]) {
	int i, n, ARRAY_BYTES;
	float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
	float sum;
	float seq_sum;
	clock_t t;

	srand(time(NULL));

	if (argc == 2) {
		n = atoi(argv[1]);
	}
	else {
		printf("N? ");
		fflush(stdout);
		scanf("%d", &n);
	}

	int BLOCKS_PER_GRID = (unsigned int)ceil(n / (float)NUM_OF_GPU_THREADS);
	printf("bpg = %d\n", BLOCKS_PER_GRID);

	// arrays n host
	ARRAY_BYTES = n * sizeof(float);
	h_A = (float *)malloc(ARRAY_BYTES);
	h_B = (float *)malloc(ARRAY_BYTES);
	h_C = (float *)malloc(BLOCKS_PER_GRID * sizeof(float));
	printf("\ncreating A and B...\n\n");
	vecFillRand(n, h_A);
	vecFillRand(n, h_B);
	vecFillRand(BLOCKS_PER_GRID, h_C);

	// arrays on device
	cudaMalloc((void**)&d_A, ARRAY_BYTES);
	cudaMalloc((void**)&d_B, ARRAY_BYTES);
	cudaMalloc((void**)&d_C, BLOCKS_PER_GRID * sizeof(float));

	// transfer the arrays to the GPU
	cudaMemcpy(d_A, h_A, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, BLOCKS_PER_GRID * sizeof(float), cudaMemcpyHostToDevice);

	// TIME START
	// create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// record time into start event
	cudaEventRecord(start, 0); // 0 is the default stream id

							   // launch the kernel
	dim3 block(NUM_OF_GPU_THREADS); // 256, 1, 1
	dim3 grid(BLOCKS_PER_GRID);
	printf("computing dotProduct... \n");
	dotProduct <<<grid, block >>>(d_A, d_B, d_C, n);

	// block until the device has completed
	cudaDeviceSynchronize();

	// check if kernel execution generated an error
	// Check for any CUDA errors
	checkCUDAError("kernel invocation");

	// TIME END
	// record time into stop event
	cudaEventRecord(stop, 0);
	// synchronize stop event to wait for end of kernel execution on stream 0
	cudaEventSynchronize(stop);
	// compute elapsed time (done by CUDA run-time)
	float elapsed_kernel = 0.f;
	cudaEventElapsedTime(&elapsed_kernel, start, stop);
	// release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// print krenel time
	printf("CUDA TIME: %f \n\n", elapsed_kernel / 1000);

	// copy back the result array to the CPU
	cudaMemcpy(h_C, d_C, BLOCKS_PER_GRID * sizeof(float), cudaMemcpyDeviceToHost);

	// Check for any CUDA errors
	checkCUDAError("memcpy");

	// compute sum
	sum = 0;
	for (i = 0; i < BLOCKS_PER_GRID; i++)
		sum += h_C[i];

	//  launch sequential
	t = clock();
	printf("computing seq_dotProduct... \n");
	seq_sum = seq_dotProduct(h_A, h_B, n);
	t = clock() - t;
	printf("SEQ TIME: %f \n\n", ((float)t) / CLOCKS_PER_SEC);

	// check sum and seq_sum
	float value = abs((sum - seq_sum) / sum);
	if (value > ACCURACY) {
		printf("Test FAILED: err: %f cpu: %f  gpu: %f \n", value, seq_sum, sum);
	}
	else {
		printf("Test PASSED \n");
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}