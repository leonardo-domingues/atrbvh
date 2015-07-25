#include "BVHTreeInstanceManager.h"

#include "CudaErrorCheck.h"

namespace BVHRT
{

BVHTreeInstanceManager::BVHTreeInstanceManager()
{
}

BVHTreeInstanceManager::~BVHTreeInstanceManager()
{
}

BVHTree* BVHTreeInstanceManager::CreateHostTree(unsigned int numberOfTriangles) const
{
    return new BVHTree(numberOfTriangles);
}

BVHTree* BVHTreeInstanceManager::CreateDeviceTree(unsigned int numberOfTriangles,
                                                  const BVHTree* hostTree) const
{
    BVHTree tempTree(numberOfTriangles, false);

    // Allocate device memory
    unsigned int numberOfElements = 2 * numberOfTriangles - 1;
    checkCudaError(cudaMalloc(&tempTree.data.parentIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.data.leftIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.data.rightIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.data.dataIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.data.boundingBoxMin, numberOfElements * sizeof(float4)));
    checkCudaError(cudaMalloc(&tempTree.data.boundingBoxMax, numberOfElements * sizeof(float4)));
    checkCudaError(cudaMalloc(&tempTree.data.area, numberOfElements * sizeof(float)));

    // Copy host data to device
    if (hostTree)
    {
        checkCudaError(cudaMemcpy(tempTree.data.parentIndices, hostTree->data.parentIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.data.leftIndices, hostTree->data.leftIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.data.rightIndices, hostTree->data.rightIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.data.dataIndices, hostTree->data.dataIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.data.boundingBoxMin, hostTree->data.boundingBoxMin,
                numberOfElements * sizeof(float4), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.data.boundingBoxMax, hostTree->data.boundingBoxMax,
                numberOfElements * sizeof(float4), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.data.area, hostTree->data.area,
                numberOfElements * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(&tempTree.data.rootIndex, &hostTree->data.rootIndex, 
                sizeof(int), cudaMemcpyHostToDevice));
    }

    // Copy temp class to device memory
    BVHTree* deviceTree;
    checkCudaError(cudaMalloc(&deviceTree, sizeof(BVHTree)));
    checkCudaError(cudaMemcpy(deviceTree, &tempTree, sizeof(BVHTree), cudaMemcpyHostToDevice));

    // Reset the temporary tree data, so arrays are not freed when its destructor is invoked
    ResetTempTree(&tempTree);

    return deviceTree;
}

BVHTree* BVHTreeInstanceManager::CreateDeviceTree(unsigned int numberOfTriangles) const
{
    return CreateDeviceTree(numberOfTriangles, nullptr);
}

BVHTree* BVHTreeInstanceManager::HostToDeviceTree(const BVHTree* hostTree) const
{
    return CreateDeviceTree(hostTree->numberOfTriangles, hostTree);
}

BVHTree* BVHTreeInstanceManager::DeviceToHostTree(const BVHTree* deviceTree) const
{
    checkCudaError(cudaDeviceSynchronize());

    BVHTree* tempTree = new BVHTree(0, false);
    checkCudaError(cudaMemcpy(tempTree, deviceTree, sizeof(BVHTree), cudaMemcpyDeviceToHost));

    BVHTree* hostTree = new BVHTree(tempTree->numberOfTriangles);

    // Copy device memory to host
    unsigned int numberOfElements = tempTree->numberOfTriangles * 2 - 1;
    checkCudaError(cudaMemcpy(hostTree->data.parentIndices, tempTree->data.parentIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->data.leftIndices, tempTree->data.leftIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->data.rightIndices, tempTree->data.rightIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->data.dataIndices, tempTree->data.dataIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->data.boundingBoxMin, tempTree->data.boundingBoxMin,
            numberOfElements * sizeof(float4), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->data.boundingBoxMax, tempTree->data.boundingBoxMax,
            numberOfElements * sizeof(float4), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->data.area, tempTree->data.area,
            numberOfElements * sizeof(float), cudaMemcpyDeviceToHost));
    hostTree->data.rootIndex = tempTree->data.rootIndex;

    // Reset the temporary tree data, so arrays are not freed when its destructor is invoked
    ResetTempTree(tempTree);

    delete tempTree;

    return hostTree;
}

void BVHTreeInstanceManager::FreeDeviceTree(BVHTree* deviceTree) const
{
    BVHTree tempTree(0, false);
    checkCudaError(cudaMemcpy(&tempTree, deviceTree, sizeof(BVHTree), cudaMemcpyDeviceToHost));

    // Free arrays from tree data
    checkCudaError(cudaFree(tempTree.data.parentIndices));
    checkCudaError(cudaFree(tempTree.data.leftIndices));
    checkCudaError(cudaFree(tempTree.data.rightIndices));
    checkCudaError(cudaFree(tempTree.data.dataIndices));
    checkCudaError(cudaFree(tempTree.data.boundingBoxMin));
    checkCudaError(cudaFree(tempTree.data.boundingBoxMax));
    checkCudaError(cudaFree(tempTree.data.area));

    // Reset the temporary tree data, so arrays are not freed when its destructor is invoked
    ResetTempTree(&tempTree);

    checkCudaError(cudaFree(deviceTree));
}

void BVHTreeInstanceManager::ResetTempTree(BVHTree* tempTree) const
{
    tempTree->data.parentIndices = nullptr;
    tempTree->data.leftIndices = nullptr;
    tempTree->data.rightIndices = nullptr;
    tempTree->data.dataIndices = nullptr;
    tempTree->data.boundingBoxMin = nullptr;
    tempTree->data.boundingBoxMax = nullptr;
    tempTree->data.area = nullptr;
}

BVHCollapsedTree* BVHTreeInstanceManager::CreateHostCollapsedTree(
        unsigned int numberOfTriangles) const
{
    return new BVHCollapsedTree(numberOfTriangles);
}

BVHCollapsedTree* BVHTreeInstanceManager::CreateDeviceCollapsedTree(
    unsigned int numberOfTriangles) const
{
    return CreateDeviceCollapsedTree(numberOfTriangles, nullptr);
}

BVHCollapsedTree* BVHTreeInstanceManager::CreateDeviceCollapsedTree(
        unsigned int numberOfTriangles, const BVHCollapsedTree* hostTree) const
{
    BVHCollapsedTree tempTree(numberOfTriangles, false);

    // Allocate device memory
    unsigned int numberOfElements = 2 * numberOfTriangles - 1;
    checkCudaError(cudaMalloc(&tempTree.nodes.parentIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.nodes.leftIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.nodes.rightIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.nodes.dataIndices, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.nodes.triangleCount, numberOfElements * sizeof(int)));
    checkCudaError(cudaMalloc(&tempTree.nodes.boundingBoxMin, numberOfElements * sizeof(float4)));
    checkCudaError(cudaMalloc(&tempTree.nodes.boundingBoxMax, numberOfElements * sizeof(float4)));
    checkCudaError(cudaMalloc(&tempTree.triangleIndices, numberOfTriangles * sizeof(int)));    

    // Copy host data to device
    if (hostTree)
    {
        checkCudaError(cudaMemcpy(tempTree.nodes.parentIndices, hostTree->nodes.parentIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.nodes.leftIndices, hostTree->nodes.leftIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.nodes.rightIndices, hostTree->nodes.rightIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.nodes.dataIndices, hostTree->nodes.dataIndices,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.nodes.triangleCount, hostTree->nodes.triangleCount,
                numberOfElements * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.nodes.boundingBoxMin, hostTree->nodes.boundingBoxMin,
                numberOfElements * sizeof(float4), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.nodes.boundingBoxMax, hostTree->nodes.boundingBoxMax,
                numberOfElements * sizeof(float4), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(tempTree.triangleIndices, hostTree->triangleIndices,
                numberOfTriangles * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(&tempTree.rootIndex, &hostTree->rootIndex, sizeof(int), 
                cudaMemcpyHostToDevice));
    }

    // Copy temp class to device memory
    BVHCollapsedTree* deviceTree;
    checkCudaError(cudaMalloc(&deviceTree, sizeof(BVHCollapsedTree)));
    checkCudaError(cudaMemcpy(deviceTree, &tempTree, sizeof(BVHCollapsedTree), cudaMemcpyHostToDevice));

    // Reset the temporary tree data, so arrays are not freed when its destructor is invoked
    ResetTempTree(&tempTree);

    return deviceTree;
}

BVHCollapsedTree* BVHTreeInstanceManager::HostToDeviceCollapsedTree(
        const BVHCollapsedTree* hostTree) const
{
    return CreateDeviceCollapsedTree(hostTree->numberOfTriangles, hostTree);
}

BVHCollapsedTree* BVHTreeInstanceManager::DeviceToHostCollapsedTree(
        const BVHCollapsedTree* deviceTree) const
{
    BVHCollapsedTree* tempTree = new BVHCollapsedTree(0, false);
    checkCudaError(cudaMemcpy(tempTree, deviceTree, sizeof(BVHCollapsedTree), 
            cudaMemcpyDeviceToHost));

    BVHCollapsedTree* hostTree = new BVHCollapsedTree(tempTree->numberOfTriangles);

    // Copy device memory to host
    unsigned int numberOfElements = tempTree->numberOfTriangles * 2 - 1;
    checkCudaError(cudaMemcpy(hostTree->nodes.parentIndices, tempTree->nodes.parentIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->nodes.leftIndices, tempTree->nodes.leftIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->nodes.rightIndices, tempTree->nodes.rightIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->nodes.dataIndices, tempTree->nodes.dataIndices,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->nodes.triangleCount, tempTree->nodes.triangleCount,
            numberOfElements * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->nodes.boundingBoxMin, tempTree->nodes.boundingBoxMin,
            numberOfElements * sizeof(float4), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->nodes.boundingBoxMax, tempTree->nodes.boundingBoxMax,
            numberOfElements * sizeof(float4), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hostTree->triangleIndices, tempTree->triangleIndices,
            tempTree->numberOfTriangles * sizeof(int), cudaMemcpyDeviceToHost));

   // hostTree->data.rootIndex = tempTree->data.rootIndex;

    // Reset the temporary tree data, so arrays are not freed when its destructor is invoked
    ResetTempTree(tempTree);

    delete tempTree;

    return hostTree;
}

void BVHTreeInstanceManager::FreeDeviceCollapsedTree(BVHCollapsedTree* deviceTree) const
{
    BVHCollapsedTree tempTree(0, false);
    checkCudaError(cudaMemcpy(&tempTree, deviceTree, sizeof(BVHCollapsedTree), 
            cudaMemcpyDeviceToHost));

    // Free arrays from tree data
    checkCudaError(cudaFree(tempTree.nodes.parentIndices));
    checkCudaError(cudaFree(tempTree.nodes.leftIndices));
    checkCudaError(cudaFree(tempTree.nodes.rightIndices));
    checkCudaError(cudaFree(tempTree.nodes.dataIndices));
    checkCudaError(cudaFree(tempTree.nodes.triangleCount));
    checkCudaError(cudaFree(tempTree.nodes.boundingBoxMin));
    checkCudaError(cudaFree(tempTree.nodes.boundingBoxMax));
    checkCudaError(cudaFree(tempTree.triangleIndices));    

    // Reset the temporary tree data, so arrays are not freed when its destructor is invoked
    ResetTempTree(&tempTree);

    checkCudaError(cudaFree(deviceTree));
}

void BVHTreeInstanceManager::ResetTempTree(BVHCollapsedTree* tempTree) const
{
    tempTree->nodes.parentIndices = nullptr;
    tempTree->nodes.leftIndices = nullptr;
    tempTree->nodes.rightIndices = nullptr;
    tempTree->nodes.dataIndices = nullptr;
    tempTree->nodes.boundingBoxMin = nullptr;
    tempTree->nodes.boundingBoxMax = nullptr;
    tempTree->nodes.triangleCount = nullptr;
    tempTree->triangleIndices = nullptr;
}

}
