#include <cuda_runtime.h>
#include <cuda.h>

void cudasetdevice_(int *device)
{
	cudaSetDevice(*device);
}

void cudagetdevicecount_(int *count)
{
	cudaGetDeviceCount(count);
}

void cudafree_(int *ptr)
{
	cudaFree((void*)*ptr);
}

void cuinit_()
{
	cuInit(0);
}
