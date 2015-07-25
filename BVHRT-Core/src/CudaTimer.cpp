#include "CudaTimer.h"

#include <cassert>

namespace BVHRT
{

CudaTimer::CudaTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaTimer::Start(cudaStream_t stream)
{
    cudaEventRecord(start, stream);
    started = true;
    stopped = false;
}

void CudaTimer::Stop(cudaStream_t stream)
{
    assert(started);
    cudaEventRecord(stop, stream);
    stopped = true;
    started = false;
}

float CudaTimer::ElapsedTime() const
{
    assert(stopped);
    if (!stopped)
    {
        return 0.0f;
    }
    float elapsedTime;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

}