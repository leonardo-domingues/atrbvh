#include "BVHTreeCollapser.h"

#include "BVHTreeInstanceManager.h"
#include "CudaErrorCheck.h"
#include "CudaTimer.h"
#include "TreeCollapser.h"

#include <iostream>

namespace BVHRT
{

BVHTreeCollapser::BVHTreeCollapser()
{
}

BVHTreeCollapser::~BVHTreeCollapser()
{
}

BVHCollapsedTree*  BVHTreeCollapser::Collapse(BVHTree* deviceTree, float* sah, float ci, float ct)
{
    // Get numberOfTriangles
    BVHTreeInstanceManager manager;
    BVHTree* hostTree = manager.DeviceToHostTree(deviceTree);
    unsigned int numberOfTriangles = hostTree->NumberOfTriangles();
    delete hostTree;

    unsigned int numberOfNodes = 2 * numberOfTriangles - 1;
    BVHCollapsedTree* deviceCollapsedTree = manager.CreateDeviceCollapsedTree(numberOfTriangles);
    float copyTreeTime, calculateCostsTime, collapseTreeTime;

    // Create atomic counters buffer
    unsigned int* deviceCounters;
    checkCudaError(cudaMalloc(&deviceCounters, (numberOfTriangles - 1) * sizeof(unsigned int)));
    
    // Allocate device memory
    float* deviceCost;
    checkCudaError(cudaMalloc(&deviceCost, numberOfNodes * sizeof(float)));
    int* deviceCollapse;
    checkCudaError(cudaMalloc(&deviceCollapse, numberOfNodes * sizeof(int)));    
    int* deviceDataPosition;
    checkCudaError(cudaMalloc(&deviceDataPosition, sizeof(int)));

    CudaTimer timer;
    timer.Start();

    copyTreeTime = DeviceCopyTree(numberOfTriangles, deviceTree, deviceCollapsedTree);

    // Initialize device memory
    checkCudaError(cudaMemset(deviceCounters, 0xFF,
        (numberOfTriangles - 1) * sizeof(unsigned int)));
    checkCudaError(cudaMemset(deviceCollapse, 0x00, numberOfNodes * sizeof(int)));

    calculateCostsTime = DeviceCalculateCosts(numberOfTriangles, deviceTree, deviceCollapsedTree, 
            deviceCounters, deviceCost, deviceCollapse, ci, ct);

    // Initialize device memory
    checkCudaError(cudaMemset(deviceCounters, 0xFF, 
            (numberOfTriangles - 1) * sizeof(unsigned int)));
    checkCudaError(cudaMemset(deviceDataPosition, 0x00, sizeof(int)));
    
    collapseTreeTime = DeviceCollapseTree(numberOfTriangles, deviceTree, deviceCollapsedTree, 
            deviceCounters, deviceCollapse, deviceDataPosition);

    timer.Stop();
    
    // Calculate SAH value
    hostTree = manager.DeviceToHostTree(deviceTree);
    float* cost = new float[numberOfNodes];
    checkCudaError(cudaMemcpy(cost, deviceCost, numberOfNodes * sizeof(float), 
            cudaMemcpyDeviceToHost));
    *sah = cost[hostTree->RootIndex()] / hostTree->Area(hostTree->RootIndex());
    delete hostTree;
    delete[] cost;

    // Free allocated memory
    checkCudaError(cudaFree(deviceDataPosition));
    checkCudaError(cudaFree(deviceCollapse));
    checkCudaError(cudaFree(deviceCost));
    checkCudaError(cudaFree(deviceCounters));

    std::cout << std::endl << "Tree collapsing" << std::endl;
    std::cout << "\tCollapse time: " << timer.ElapsedTime() << " ms" << std::endl;
    std::cout << "\tSAH: " << *sah << std::endl;
    std::cout << "\tKernel execution times:" << std::endl;
    std::cout << "\t               Copy tree: " << copyTreeTime << " ms" << std::endl;
    std::cout << "\t         Calculate costs: " << calculateCostsTime << " ms" << std::endl;
    std::cout << "\t           Collapse tree: " << collapseTreeTime << " ms" << std::endl;
    std::cout << std::endl;

    return deviceCollapsedTree;
}

}
