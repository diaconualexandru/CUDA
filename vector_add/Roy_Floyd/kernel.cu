#include<cuda.h>
#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int main(void) {
	void Roy_Floyd(int *, int);
	const int Width = 6;
	int a[Width*Width] = { 0, 2, 5, 999, 999, 999, 999, 0, 7, 1, 999 , 8, 999, 999, 0, 4, 999, 999, 999, 999, 999, 0, 3, 999, 999, 999, 2, 999, 0, 3, 999, 5, 999, 2, 4, 0 };
	Roy_Floyd(a, Width);
	for (int i = 0; i < (Width*Width); i++) {
		printf("%d \t", a[i]);
		if ((i + 1) % Width == 0) { printf("\n"); }
	}
	int quit;
	scanf("%d", &quit);
	return 0;
}

//Matrix multiplication kernel - thread specification
__global__ void Compute_Path(int *Md, int Width, int k) {
	//2D Thread ID
	int ROW = blockIdx.x;
	int COL = threadIdx.x;

	float tmpSum = 0;
	//Pvalue stores the Pd element that is computed by the thread
	
		if (Md[ROW * Width + COL] > Md[ROW * Width + k] + Md[k * Width + COL])
			Md[ROW * Width + COL] = Md[ROW * Width + k] + Md[k * Width + COL];
}

void Roy_Floyd(int *M, int Width) {
	int size = Width*Width * sizeof(int);
	int *Md;

	//Transfer M and N to device memory
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);

	//Launch the device computation threads!
	for (int k = 0; k < Width; k++)
		Compute_Path << <Width, Width >> >(Md, Width, k);

	//Transfer P from device to host
	cudaMemcpy(M, Md, size, cudaMemcpyDeviceToHost);

	//Free device matrices
	cudaFree(Md);
}