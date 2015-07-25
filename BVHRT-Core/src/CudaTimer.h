#pragma once

#include <cuda_runtime_api.h>

namespace BVHRT
{

/// <summary> A timer based on CUDA events. One should be careful when using this timer to measure 
///           host code. If that is the case, be sure to supply the Start() method with an empty 
///           stream. </summary>
///
/// <remarks> Leonardo, 12/28/2014. </remarks>
class CudaTimer
{
public:

    /// <summary> Default constructor. </summary>
    ///
    /// <remarks> Leonardo, 12/28/2014. </remarks>
    CudaTimer();

    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 12/28/2014. </remarks>
    ~CudaTimer();

    /// <summary> Starts the timer. Does not block the CPU. </summary>
    ///
    /// <remarks> Leonardo, 12/28/2014. </remarks>
    ///
    /// <param name="stream"> [out] CUDA stream the timer is based on. If none is given, the 
    ///                       default stream is used. </param>
    void Start(cudaStream_t stream = 0);

    /// <summary> Stops the timer. Does not block the CPU. </summary>
    ///
    /// <remarks> Leonardo, 12/28/2014. </remarks>
    ///
    /// <param name="stream"> [out] CUDA stream the timer is based on. If none is given, the 
    ///                       default stream is used. </param>
    void Stop(cudaStream_t stream = 0);

    /// <summary> Get the elapsed time between the Start() and Stop() calls. This methods blocks 
    ///           the CPU until all kernels that were launched between the Start() and Stop() 
    ///           calls in the specified stream finish executing.</summary>
    ///
    /// <remarks> Leonardo, 12/28/2014. </remarks>
    ///
    /// <returns> The elapsed time, in milliseconds. </returns>
    float ElapsedTime() const;

private:
    cudaEvent_t start, stop;
    bool started, stopped;
};
}
