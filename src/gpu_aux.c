#include <cuda_runtime.h>

void cudasetdevice_(int *device)
{
	cudaSetDevice(*device);
}

void cudagetdevicecount_(int *count)
{
	cudaGetDeviceCount(count);
}
