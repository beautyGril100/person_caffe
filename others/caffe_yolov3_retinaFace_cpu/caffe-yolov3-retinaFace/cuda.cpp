/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04
 */
#include "cuda.h"
#include "blas.h"

#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

void error(const char* s)
{
    perror(s);
    assert(0);
    exit(-1);
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

float* cuda_make_array(float* x,size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
#if 0
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
#else
    x_gpu = (float*)calloc(n,sizeof(float));
    if(x){
        memcpy(x_gpu, x, size);
    }
#endif
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float* x_gpu)
{
#if 0
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
#else
    free(x_gpu);
    x_gpu = NULL;
#endif
}

void cuda_push_array(float *x_gpu,float* x,size_t n)
{
    size_t size = sizeof(float)*n;
 #if 0
    cudaError_t status = cudaMemcpy(x_gpu,x,size,cudaMemcpyHostToDevice);
    check_error(status);
 #else
    memcpy(x_gpu, x, size);
 #endif
}



void cuda_pull_array(float *x_gpu,float* x,size_t n)
{
    size_t size = sizeof(float)*n;
#if 0
    cudaError_t status = cudaMemcpy(x,x_gpu,size,cudaMemcpyDeviceToHost);
    check_error(status);
#else
    memcpy(x, x_gpu, size);
#endif
}
