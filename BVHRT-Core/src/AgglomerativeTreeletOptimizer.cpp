#include "AgglomerativeTreeletOptimizer.h"

#include "AgglomerativeScheduler.h"
#include "AgglomerativeTreelet.h"
#include "BVHTreeInstanceManager.h"
#include "Commons.cuh"
#include "CudaErrorCheck.h"

#define MAX_TREELET_SIZE_DIST_SHARED_MEM 20

namespace BVHRT
{

AgglomerativeTreeletOptimizer::AgglomerativeTreeletOptimizer(int treeletSize, int iterations) :
        treeletSize(treeletSize), iterations(iterations)
{
}


AgglomerativeTreeletOptimizer::~AgglomerativeTreeletOptimizer()
{
}

void AgglomerativeTreeletOptimizer::Optimize(BVHTree* deviceTree)
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

    // Allocate schedule
    BVHRT::AgglomerativeScheduler scheduler;
    std::vector<int> schedule;
    int scheduleSize;
    scheduler.GenerateScheduleLower(treeletSize, warpSize, schedule, scheduleSize);
    int* deviceSchedule;
    checkCudaError(cudaMalloc(&deviceSchedule, scheduleSize * sizeof(int)));
    globalMemoryUsed += scheduleSize * sizeof(int);
    checkCudaError(cudaMemcpy(deviceSchedule, schedule.data(), scheduleSize * sizeof(int),
            cudaMemcpyHostToDevice));

    // Allocate distance matrix
    int numberOfElements = sumArithmeticSequence(treeletSize - 1, 1, treeletSize - 1);
    float* deviceDistanceMatrix = nullptr;    
    if (treeletSize > MAX_TREELET_SIZE_DIST_SHARED_MEM)
    {
        int numberOfWarps = ((numberOfTriangles + warpSize - 1) / warpSize);
        checkCudaError(cudaMalloc(&deviceDistanceMatrix,
                numberOfWarps * numberOfElements * sizeof(float)));
        globalMemoryUsed += numberOfWarps * numberOfElements * sizeof(float);
    }

    int gamma = treeletSize;
    float elapsedTime = 0;
    for (int i = 0; i < iterations; ++i)
    {
        checkCudaError(cudaMemset(deviceCounters, 0x00, 
                (numberOfTriangles - 1) * sizeof(unsigned int)));
        if (treeletSize > MAX_TREELET_SIZE_DIST_SHARED_MEM)
        {
            elapsedTime += DeviceAgglomerativeTreeletOptimizer(numberOfTriangles, deviceTree,
                deviceCounters, deviceSubtreeTrianglesCount, treeletSize, gamma, deviceSchedule,
                scheduleSize, deviceDistanceMatrix, numberOfElements, deviceNodesSah);
        }
        else
        {
            elapsedTime += DeviceAgglomerativeSmallTreeletOptimizer(numberOfTriangles, deviceTree,
                deviceCounters, deviceSubtreeTrianglesCount, treeletSize, gamma, deviceSchedule,
                scheduleSize, deviceNodesSah);
        }
        gamma *= 2;
    }

    // Free device memory
    checkCudaError(cudaFree(deviceCounters));
    checkCudaError(cudaFree(deviceSubtreeTrianglesCount));
    checkCudaError(cudaFree(deviceNodesSah));
    checkCudaError(cudaFree(deviceSchedule));
    checkCudaError(cudaFree(deviceDistanceMatrix));

    // Print report
    hostTree = manager.DeviceToHostTree(deviceTree);
    std::cout << std::endl << "Agglomerative BVH" << std::endl;
    std::cout << "\tTreelet size: " << treeletSize << std::endl;
    std::cout << "\tIterations: " << iterations << std::endl;
    std::cout << "\tOptimize time: " << elapsedTime << " ms" << std::endl;
    std::cout << "\tSAH: " << hostTree->SAH() << std::endl;
    std::cout << "\tGlobal memory used: " << globalMemoryUsed << " B" << std::endl << std::endl;
    delete hostTree;
}

}