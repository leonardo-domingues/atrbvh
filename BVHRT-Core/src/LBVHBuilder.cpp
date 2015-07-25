#include "LBVHBuilder.h"

#include "BVHTreeInstanceManager.h"
#include "CubWrapper.h"
#include "CudaErrorCheck.h"
#include "CudaTimer.h"
#include "Defines.h"
#include "LBVH.h"

namespace BVHRT
{

LBVHBuilder::LBVHBuilder(bool use64BitMortonCodes) : use64BitMortonCodes(use64BitMortonCodes)
{
}

LBVHBuilder::~LBVHBuilder()
{
}

BVHTree* LBVHBuilder::BuildTree(const SceneWrapper* sceneWrapper)
{
    BVHTree* tree = nullptr;
    if (use64BitMortonCodes)
    {
        tree = BuildTreeNoTriangleSplitting<unsigned long long int>(sceneWrapper);
    }
    else
    {
        tree = BuildTreeNoTriangleSplitting<unsigned int>(sceneWrapper);
    }

    return tree;
}

template <typename T> BVHTree* LBVHBuilder::BuildTreeNoTriangleSplitting(
        const SceneWrapper* sceneWrapper) const
{
    CudaTimer timer;
    timer.Start();

    float timeMortonCodes, timeSort, timeBuildTree, timeBoundingBoxes;

    const Scene* deviceScene = sceneWrapper->DeviceScene();
    unsigned int numberOfTriangles = sceneWrapper->HostScene()->numberOfTriangles;

    size_t globalMemoryUsed = 0;

    // Create Morton codes buffer
    T* deviceMortonCodes;
    checkCudaError(cudaMalloc(&deviceMortonCodes, numberOfTriangles * sizeof(T)));
    globalMemoryUsed += numberOfTriangles * sizeof(T);

    // Array of indices that will be used in the sort algorithm
    unsigned int* deviceIndices;
    checkCudaError(cudaMalloc(&deviceIndices, numberOfTriangles * sizeof(unsigned int)));
    globalMemoryUsed += numberOfTriangles * sizeof(unsigned int);

    // Generate Morton codes
    timeMortonCodes = DeviceGenerateMortonCodes(numberOfTriangles, deviceScene, deviceMortonCodes,
            deviceIndices);

    // Create sorted keys buffer
    T* deviceSortedMortonCodes;
    checkCudaError(cudaMalloc(&deviceSortedMortonCodes, numberOfTriangles * sizeof(T)));
    globalMemoryUsed += numberOfTriangles * sizeof(T);

    // Create sorted indices buffer
    unsigned int* deviceSortedIndices;
    checkCudaError(cudaMalloc(&deviceSortedIndices, numberOfTriangles * sizeof(unsigned int)));
    globalMemoryUsed += numberOfTriangles * sizeof(unsigned int);

    // Sort Morton codes
    timeSort = DeviceSort(numberOfTriangles, &deviceMortonCodes, &deviceSortedMortonCodes,
            &deviceIndices, &deviceSortedIndices);

    // Free device memory
    checkCudaError(cudaFree(deviceMortonCodes));
    checkCudaError(cudaFree(deviceIndices));

    // Create BVH instance
    BVHTreeInstanceManager factory;
    SoABVHTree* deviceTree = factory.CreateDeviceTree(numberOfTriangles);

    // Create BVH
    timeBuildTree = DeviceBuildTree(numberOfTriangles, deviceSortedMortonCodes, 
            deviceSortedIndices, deviceTree);

    // Create atomic counters buffer
    unsigned int* deviceCounters;
    checkCudaError(cudaMalloc(&deviceCounters, (numberOfTriangles - 1) * sizeof(unsigned int)));
    checkCudaError(cudaMemset(deviceCounters, 0xFF, 
            (numberOfTriangles - 1) * sizeof(unsigned int)));
    globalMemoryUsed += (numberOfTriangles - 1) * sizeof(unsigned int);

    // Calculate BVH nodes bounding boxes
    timeBoundingBoxes = DeviceCalculateNodeBoundingBoxes(numberOfTriangles, deviceScene, 
            deviceTree, deviceCounters);

    // Free device memory
    checkCudaError(cudaFree(deviceCounters));
    checkCudaError(cudaFree(deviceSortedMortonCodes));
    checkCudaError(cudaFree(deviceSortedIndices));

    timer.Stop();

    // Print report
    BVHTree* hostTree = factory.DeviceToHostTree(deviceTree);
    if (use64BitMortonCodes)
    {
        std::cout << std::endl << "LBVH64" << std::endl;
    }
    else
    {
        std::cout << std::endl << "LBVH32" << std::endl;
    }    
    std::cout << "\tBuild time: " << timer.ElapsedTime() << " ms" << std::endl;
    std::cout << "\tSAH: " << hostTree->SAH() << std::endl;
    std::cout << "\tKernel execution times:" << std::endl;
    std::cout << "\t  Calculate Morton codes: " << timeMortonCodes << " ms" << std::endl;
    std::cout << "\t       Sort Morton codes: " << timeSort << " ms" << std::endl;
    std::cout << "\t              Build tree: " << timeBuildTree << " ms" << std::endl;
    std::cout << "\tCalculate bounding boxes: " << timeBoundingBoxes << " ms" << std::endl;
    std::cout << "\tGlobal memory used: " << globalMemoryUsed << " B" << std::endl << std::endl;
    delete hostTree;

    checkCudaError(cudaGetLastError());

    return deviceTree;
}

}
