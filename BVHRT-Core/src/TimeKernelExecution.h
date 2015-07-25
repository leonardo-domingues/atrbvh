#pragma once

#include "Defines.h"

/// <summary> Measure the execution time of a kernel call using CUDA events. </summary>
///
/// <remarks> Leonardo, 12/28/2014. </remarks>
///
/// <typeparam name="KernelCall"> A lambda function containing the kernel call. </typeparam>
/// <param name="kernel"> The kernel call. </param>
/// <param name="stream"> The stream the kernel will run on. </param>
///
/// <returns> The kernel execution time, in milliseconds. </returns>
template<typename KernelCall> float TimeKernelExecution(KernelCall kernel, cudaStream_t stream = 0)
{
    float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
#endif
    kernel();
#ifdef MEASURE_EXECUTION_TIMES
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    return elapsedTime;
}
