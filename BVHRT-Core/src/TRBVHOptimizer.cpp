#include "TRBVHOptimizer.h"

#include "BVHTreeInstanceManager.h"
#include "CudaErrorCheck.h"
#include "TRBVH.h"
#include "TRBVHScheduler.h"
#include <vector>

namespace BVHRT
{

TRBVHOptimizer::TRBVHOptimizer(int treeletSize, int iterations) : treeletSize(treeletSize), 
        iterations(iterations)
{
}

TRBVHOptimizer::~TRBVHOptimizer()
{
}

void TRBVHOptimizer::Optimize(BVHTree* deviceTree)
{
    BVHTreeInstanceManager manager;
    BVHTree* hostTree = manager.DeviceToHostTree(deviceTree);
    unsigned int numberOfTriangles = hostTree->NumberOfTriangles();
    delete hostTree;

    size_t globalMemoryUsed = 0;

    // Allocate device memory
    unsigned int* deviceCounters;
    checkCudaError(cudaMalloc(&deviceCounters, (numberOfTriangles - 1) * sizeof(unsigned int)));
    // Counters are already accounted for during LBVH. Commented so as to not count them twice on 
    // the execution script
    //globalMemoryUsed += (numberOfTriangles - 1) * sizeof(unsigned int);

    int* deviceSubtreeTrianglesCount;
    checkCudaError(cudaMalloc(&deviceSubtreeTrianglesCount, 
            (2 * numberOfTriangles - 1) * sizeof(int)));
    globalMemoryUsed += (2 * numberOfTriangles - 1) * sizeof(int);

    float* deviceNodesSah;
    checkCudaError(cudaMalloc(&deviceNodesSah, (2 * numberOfTriangles - 1) * sizeof(float)));
    globalMemoryUsed += (2 * numberOfTriangles - 1) * sizeof(float);

    const int warpSize = 32;
    int numberOfWarps = ((numberOfTriangles + warpSize - 1) / warpSize);
    size_t size = (1 << treeletSize) * numberOfWarps;

    float4* deviceBoundingBoxesMin;
    checkCudaError(cudaMalloc(&deviceBoundingBoxesMin, size * sizeof(float4)));
    globalMemoryUsed += size * sizeof(float4);

    float4* deviceBoundingBoxesMax;
    checkCudaError(cudaMalloc(&deviceBoundingBoxesMax, size * sizeof(float4)));
    globalMemoryUsed += size * sizeof(float4);

    float* deviceSubsetAreas;
    checkCudaError(cudaMalloc(&deviceSubsetAreas, size * sizeof(float)));
    globalMemoryUsed += size * sizeof(float);

    int* deviceStackNode;
    checkCudaError(cudaMalloc(&deviceStackNode, (treeletSize - 1) * numberOfWarps * sizeof(int)));
    globalMemoryUsed += (treeletSize - 1) * numberOfWarps * sizeof(int);

    char* deviceStackMask;
    checkCudaError(cudaMalloc(&deviceStackMask, (treeletSize - 1) * numberOfWarps * sizeof(char)));
    globalMemoryUsed += (treeletSize - 1) * numberOfWarps * sizeof(char);

    int* deviceStackSize;
    checkCudaError(cudaMalloc(&deviceStackSize, numberOfWarps * sizeof(int)));
    globalMemoryUsed += numberOfWarps * sizeof(int);

    int* deviceCurrentInternalNode;
    checkCudaError(cudaMalloc(&deviceCurrentInternalNode, numberOfWarps * sizeof(int)));
    globalMemoryUsed += numberOfWarps * sizeof(int);
    
    // Allocate schedule
    TRBVHScheduler scheduler;
    std::vector<std::vector<int>> schedule;
    scheduler.GenerateSchedule(treeletSize, warpSize, schedule);
    int* deviceSchedule;
    checkCudaError(cudaMalloc(&deviceSchedule, schedule.size() * warpSize * sizeof(int)));
    globalMemoryUsed += schedule.size() * warpSize * sizeof(int);
    for (unsigned int i = 0; i < schedule.size(); ++i)
    {
        checkCudaError(cudaMemcpy(deviceSchedule + warpSize * i, schedule[i].data(),
                warpSize * sizeof(int), cudaMemcpyHostToDevice));
    }

    int gamma = treeletSize;
    float elapsedTime = 0.0f;
    for (int i = 0; i < iterations; ++i)
    {
        checkCudaError(cudaMemset(deviceCounters, 0x00, (numberOfTriangles - 1) * 
                sizeof(unsigned int)));
        float time = DeviceTreeletReestructureOptimizer(numberOfTriangles, deviceTree, 
                deviceCounters, deviceSubtreeTrianglesCount, deviceNodesSah, treeletSize, gamma, 
                deviceSchedule, static_cast<int>(schedule.size()), deviceBoundingBoxesMin, 
                deviceBoundingBoxesMax, deviceSubsetAreas, deviceStackNode, deviceStackMask, 
                deviceStackSize, deviceCurrentInternalNode);

        elapsedTime += time;

        gamma *= 2;
    }

    // Free device memory
    checkCudaError(cudaFree(deviceCounters));
    checkCudaError(cudaFree(deviceSubtreeTrianglesCount));
    checkCudaError(cudaFree(deviceNodesSah));    
    checkCudaError(cudaFree(deviceSchedule));
    checkCudaError(cudaFree(deviceBoundingBoxesMin));
    checkCudaError(cudaFree(deviceBoundingBoxesMax));
    checkCudaError(cudaFree(deviceSubsetAreas));
    checkCudaError(cudaFree(deviceStackNode));
    checkCudaError(cudaFree(deviceStackMask));
    checkCudaError(cudaFree(deviceStackSize));
    checkCudaError(cudaFree(deviceCurrentInternalNode));

    // Print report
    hostTree = manager.DeviceToHostTree(deviceTree);
    std::cout << std::endl << "TRBVH" << std::endl;
    std::cout << "\tOptimize time: " << elapsedTime << " ms" << std::endl;
    std::cout << "\tSAH: " << hostTree->SAH() << std::endl;
    std::cout << "\tGlobal memory used: " << globalMemoryUsed << " B" << std::endl << std::endl;
    delete hostTree;
}

}
