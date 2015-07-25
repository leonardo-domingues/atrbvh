#pragma once

#include <cuda_runtime.h>
#include <iostream>

/// <summary> A macro that checks the returned value of functions from the CUDA API to find out if
///           any error occurred. Do not use this macro inside .cu files. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="ans"> The operation result code. </param>
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d (%d)\n", cudaGetErrorString(code), file, line, code);
        if (abort)
        {
            exit(code);
        }
    }
}