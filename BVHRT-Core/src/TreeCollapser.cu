#include "TreeCollapser.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "BVHCollapsedTree.h"
#include "Commons.cuh"
#include "TimeKernelExecution.h"

namespace BVHRT
{

__global__ void copyTreeKernel(unsigned int numberOfTriangles, BVHTree* source, 
        BVHCollapsedTree* destination)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (i >= (2 * numberOfTriangles - 1))
    {
        return;
    }

    destination->nodes.boundingBoxMax[i] = source->BoundingBoxMax(i);
    destination->nodes.boundingBoxMin[i] = source->BoundingBoxMin(i);
    destination->nodes.dataIndices[i] = source->DataIndex(i);
    destination->nodes.leftIndices[i] = source->LeftIndex(i);
    destination->nodes.rightIndices[i] = source->RightIndex(i);
    destination->nodes.parentIndices[i] = source->ParentIndex(i);
}

__global__ void calculateCostKernel(unsigned int numberOfTriangles, BVHTree* source, 
        BVHCollapsedTree* destination, unsigned int* counters, float* cost, int* collapse, 
        float ci, float ct)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (i >= numberOfTriangles)
    {
        return;
    }

    int index = i + numberOfTriangles - 1;

    // Set triangle count
    destination->nodes.triangleCount[index] = 1;

    // Calculate cost
    cost[index] = ct * calculateBoundingBoxSurfaceArea(source->BoundingBoxMin(index), 
            source->BoundingBoxMax(index));

    // Set collapse flag
    collapse[index] = 1;

    int current = source->ParentIndex(index);
    while (current != -1)
    {
        unsigned int childThreadId = atomicExch(&counters[current], i);

        // The first thread to reach a node will just die
        if (childThreadId == 0xFFFFFFFF)
        {
            return;
        }

        int left = source->LeftIndex(current);
        int right = source->RightIndex(current);

        // Calculate triangle count
        int count = destination->nodes.triangleCount[left] + 
                destination->nodes.triangleCount[right];
        destination->nodes.triangleCount[current] = count;

        // Calculate cost and check if nodes need to be collapsed
        float area = calculateBoundingBoxSurfaceArea(source->BoundingBoxMin(current), 
                source->BoundingBoxMax(current));
        cost[current] = ci * area + cost[left] + cost[right];
        float costAsLeaf = ct * area * count;
        if (costAsLeaf < cost[current])
        {
            cost[current] = costAsLeaf;
            collapse[current] = 1;
        }

        // Internal nodes have a data index of -1. This can be used to identify internal nodes 
        // from leaves
        destination->nodes.dataIndices[current] = -1;

        // Update current node pointer
        current = source->ParentIndex(current);
    }
}

__device__ void writeSubtreeLeavesNonrecursive(unsigned int numberOfTriangles, BVHTree* tree, 
        int index, int* leaves, int numberOfLeaves, int* position)
{
    // Setup stack
    leaves[numberOfLeaves - 1] = tree->LeftIndex(index);
    leaves[numberOfLeaves - 2] = tree->RightIndex(index);
    int stackSize = 2;

    while (stackSize > 0)
    {
        // Pop stack
        index = leaves[numberOfLeaves - stackSize];
        --stackSize;

        if (isLeaf(index, numberOfTriangles))
        {
            int dataIndex = tree->DataIndex(index);

            // Check if triangle is not already on the leaf
            bool unique = true;
            for (int i = 0; i < *position; ++i)
            {
                if (leaves[i] == dataIndex)
                {
                    unique = false;
                    break;
                }
            }

            if (unique)
            {
                leaves[(*position)++] = dataIndex;
            }
        }
        else
        {
            int left = tree->LeftIndex(index);
            int right = tree->RightIndex(index);
            leaves[numberOfLeaves - 1 - stackSize++] = left;
            leaves[numberOfLeaves - 1 - stackSize++] = right;
        }
    }
}

__device__ void writeSubtreeLeaves(unsigned int numberOfTriangles, BVHTree* tree, int index, 
        int* leaves, int* position)
{
    if (isLeaf(index, numberOfTriangles))
    {
        int dataIndex = tree->DataIndex(index);

        // Check if triangle is not already on the leaf
        bool unique = true;
        for (int i = 0; i < *position; ++i)
        {
            if (leaves[i] == dataIndex)
            {
                unique = false;
                break;
            }
        }

        if (unique)
        {
            leaves[(*position)++] = dataIndex;
        }

        return;
    }
    else
    {
        int left = tree->LeftIndex(index);
        int right = tree->RightIndex(index);
        writeSubtreeLeaves(numberOfTriangles, tree, left, leaves, position);
        writeSubtreeLeaves(numberOfTriangles, tree, right, leaves, position);
    }
}

__global__ void collapseTreeKernel(unsigned int numberOfTriangles, BVHTree* source,
        BVHCollapsedTree* destination, unsigned int* counters, int* collapse, int* dataPosition)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (i >= numberOfTriangles)
    {
        return;
    }

    int index = i + numberOfTriangles - 1;
    int current = source->ParentIndex(index);
    while (current != -1)
    {
        // In the counters array, we have stored the id of the thread that processed the other 
        // children of this node
        unsigned int childThreadId = atomicExch(&counters[current], i);

        // The first thread to reach a node will just die
        if (childThreadId == 0xFFFFFFFF)
        {
            return;
        }

        // Collapse children
        if (collapse[current] == 0)
        {
            int left = source->LeftIndex(current);
            int right = source->RightIndex(current);

            int leftCount = destination->nodes.triangleCount[left];
            int rightCount = destination->nodes.triangleCount[right];

            // Collapse left subtree
            if (leftCount > 1)
            {
                int offset = atomicAdd(dataPosition, leftCount);
                destination->nodes.dataIndices[left] = offset;
                int* buffer = destination->triangleIndices + offset;
                int position = 0;
                writeSubtreeLeavesNonrecursive(numberOfTriangles, source, left, buffer, leftCount,
                    &position);
                destination->nodes.triangleCount[left] = position;
            }

            // Collapse right subtree
            if (rightCount > 1)
            {
                int offset = atomicAdd(dataPosition, rightCount);
                destination->nodes.dataIndices[right] = offset;
                int* buffer = destination->triangleIndices + offset;
                int position = 0;
                writeSubtreeLeavesNonrecursive(numberOfTriangles, source, right, buffer,
                    rightCount, &position);
                destination->nodes.triangleCount[right] = position;
            }

            return;
        }

        // Update current node pointer
        current = source->ParentIndex(current);
    }
}

float DeviceCopyTree(unsigned int numberOfTriangles, BVHTree* source, 
        BVHCollapsedTree* destination)
{
    unsigned int numberOfElements = 2 * numberOfTriangles - 1;

    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numberOfElements + (blockSize.x - 1)) / blockSize.x, 1, 1);

    cudaFuncSetCacheConfig(copyTreeKernel, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        copyTreeKernel<<<gridSize, blockSize>>>(numberOfTriangles, source, destination);
    });
}

float DeviceCalculateCosts(unsigned int numberOfTriangles, BVHTree* source,
        BVHCollapsedTree* destination, unsigned int* counters, float* cost, int* collapse, 
        float ci, float ct)
{
    dim3 blockSize(512, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);

    cudaFuncSetCacheConfig(calculateCostKernel, cudaFuncCachePreferL1);
    
    return TimeKernelExecution([&]()
    {
        calculateCostKernel<<<gridSize, blockSize>>>(numberOfTriangles, source, destination, 
                counters, cost, collapse, ci, ct);
    });
}

float DeviceCollapseTree(unsigned int numberOfTriangles, BVHTree* source,
        BVHCollapsedTree* destination, unsigned int* counters, int* collapse, int* dataPosition)
{
    // Collapse tree
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);

    cudaFuncSetCacheConfig(collapseTreeKernel, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        collapseTreeKernel<<<gridSize, blockSize>>>(numberOfTriangles, source, destination, 
                counters, collapse, dataPosition);
    });   
}

}
