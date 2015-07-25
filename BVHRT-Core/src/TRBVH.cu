#include "TRBVH.h"

#include <cuda_runtime_api.h>
#include <math_functions.h>
#include <device_launch_parameters.h>

#include "Commons.cuh"
#include "TimeKernelExecution.h"
#include "Treelet.cuh"

#include <cfloat>

#define WARP_SIZE 32
#define GLOBAL_WARP_INDEX static_cast<int>((threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE)
#define WARP_INDEX static_cast<int>(threadIdx.x / WARP_SIZE)
#define WARP_ARRAY(source, elementsPerWarp) ((source) + WARP_INDEX * (elementsPerWarp))
#define WARP_ARRAY_INDEX(index, elementsPerWarp) (WARP_INDEX * (elementsPerWarp) + (index))
#define THREAD_WARP_INDEX (threadIdx.x & (WARP_SIZE - 1))

namespace BVHRT
{

__device__ void reduceOptimal(float& optimalCost, int& optimalMask, int numberOfValues)
{
    for (int i = numberOfValues >> 1; i > 0; i = (i >> 1))
    {
        float otherValue = __shfl_down(optimalCost, i);
        int otherMask = __shfl_down(optimalMask, i);
        if (otherValue < optimalCost)
        {
            optimalCost = otherValue;
            optimalMask = otherMask;
        }
    }    
}

__device__ void calculateSubsetSurfaceAreas(int treeletSize, BVHTree* tree, int* treeletLeaves,
        float* subsetAreas, float4* boundingBoxesMin, float4* boundingBoxesMax, float* costs)
{
    float bbMin[3], bbMax[3];
    if (THREAD_WARP_INDEX < treeletSize)
    {
        floatArrayFromFloat4(tree->BoundingBoxMin(treeletLeaves[THREAD_WARP_INDEX]), bbMin);
        floatArrayFromFloat4(tree->BoundingBoxMax(treeletLeaves[THREAD_WARP_INDEX]), bbMax);
    }

    // The 5 most significative bits are common ammong the thread's subsets
    int subset = THREAD_WARP_INDEX * 4;
    float3 baseMin, baseMax;
    baseMin.x = FLT_MAX;
    baseMin.y = FLT_MAX;
    baseMin.z = FLT_MAX;
    baseMax.x = -FLT_MAX;
    baseMax.y = -FLT_MAX;
    baseMax.z = -FLT_MAX;
    for (int i = (treeletSize - 5); i < treeletSize; ++i)
    {
        float3 leafBbMin, leafBbMax;
        SHFL_FLOAT3(leafBbMin, bbMin, i);
        SHFL_FLOAT3(leafBbMax, bbMax, i);
        if (subset & (1 << i))
        {
            expandBoundingBox(baseMin, baseMax, leafBbMin, leafBbMax);
        }
    }

    int iterations = max(1, 1 << (treeletSize - 5)); // Num elements / 32, rounded up
    for (int j = 0; j < iterations; ++j)
    {
        float3 subsetMin, subsetMax;
        subsetMin.x = baseMin.x;
        subsetMin.y = baseMin.y;
        subsetMin.z = baseMin.z;
        subsetMax.x = baseMax.x;
        subsetMax.y = baseMax.y;
        subsetMax.z = baseMax.z;
        for (int i = 0; i < (treeletSize - 5); ++i)
        {
            float3 leafBbMin, leafBbMax;
            SHFL_FLOAT3(leafBbMin, bbMin, i);
            SHFL_FLOAT3(leafBbMax, bbMax, i);
            if (subset & (1 << i))
            {
                expandBoundingBox(subsetMin, subsetMax, leafBbMin, leafBbMax);
            }
        }

        // Store bounding boxes and their surface areas
        int position = (1 << treeletSize) * GLOBAL_WARP_INDEX + subset;
        boundingBoxesMin[position] = float4FromFloat3(subsetMin);
        boundingBoxesMax[position] = float4FromFloat3(subsetMax);
        float subsetArea = calculateBoundingBoxSurfaceArea(subsetMin, subsetMax);
        subsetAreas[position] = subsetArea;
        costs[subset] = subsetArea;
        
        ++subset;
    }
}

__device__ void processSchedule(int numberOfRounds, int* schedule, float* costs, 
        char* partitionMasks, int treeletTriangles, float ci, float ct)
{
    for (int j = 0; j < numberOfRounds; ++j)
    {
        int subset = schedule[THREAD_WARP_INDEX + j * WARP_SIZE];
        if (subset != 0)
        {
            // Process all possible partitions of the subset
            float optimalCost = FLT_MAX;
            int optimalPartition = 0;
            int delta = (subset - 1) & subset;
            int partition = (-delta) & subset;
            float partitionCost;
            while (partition != 0)
            {
                partitionCost = costs[partition] + costs[partition ^ subset];
                if (partitionCost < optimalCost)
                {
                    optimalCost = partitionCost;
                    optimalPartition = partition;
                }
                partition = (partition - delta) & subset;
            }

            // Calculate subset SAH. Keep whichever has a lower SAH between collapsing 
            // the subset treelet or leaving it as is
            costs[subset] = min(ci * costs[subset] + optimalCost,
                    ct * costs[subset] * treeletTriangles);
            partitionMasks[subset] = static_cast<char>(optimalPartition);
        }

        WARP_SYNC;
    }
}

__device__ void processSubsets(int treeletSize, int treeletTriangles, float* costs, 
        char* partitionMasks, float ci, float ct)
{
    // Process subsets of size treeletSize-1. Each 4 threads will process a subset. There 
    // are treeletSize subsets.
    if (THREAD_WARP_INDEX < 4 * treeletSize)
    {
        // To find the nth subset of treeletSize-1 elements, start with a sequence of 
        // treeletSize ones and set the nth bit to 0
        int subset = ((1 << treeletSize) - 1) & (~(1 << (THREAD_WARP_INDEX / 4)));

        // To assemble the partitions of nth subset of size treeletSize-1, we create a 
        // mask to split that subset before the (n-1)th least significant bit. We then 
        // get the left part of the masked base number and shift left by one (thus adding 
        // the partition's 0). The last step is to OR the result with the right part of 
        // the masked number and shift one more time to set the least significant bit to 
        // 0. Below is an example for a treelet size of 7:
        // subset = 1110111 (7 bits)
        // base = abcde (5 bits)
        // partition = abc0de0 (7 bits)
        // The cast to int is required so max does not return the wrong value
        int leftMask = -(1 << max(static_cast<int>((THREAD_WARP_INDEX / 4) - 1), 0));
        // x & 3 == x % 4
        int partitionBase = (THREAD_WARP_INDEX & 3) + 1;        
        float optimalCost = FLT_MAX;
        int optimalPartition = 0;
        int numberOfPartitions = (1 << (treeletSize - 2)) - 1; 
        int partition = (((partitionBase & leftMask) << 1) |
                (partitionBase & ~leftMask)) << 1;
        for (int j = (THREAD_WARP_INDEX & 3); j < numberOfPartitions; j += 4)
        {
            float partitionCost = costs[partition] + costs[partition ^ subset];
            if (partitionCost < optimalCost)
            {
                optimalCost = partitionCost;
                optimalPartition = partition;
            }

            partitionBase += 4;
            partition = (((partitionBase & leftMask) << 1) |
                    (partitionBase & ~leftMask)) << 1;
        }

        reduceOptimal(optimalCost, optimalPartition, 4);

        if ((THREAD_WARP_INDEX & 3) == 0)
        {
            // Calculate subset SAH. Keep whichever has a lower SAH between collapsing 
            // the subset treelet or leaving it as is
            costs[subset] = min(ci * costs[subset] + optimalCost,
                    ct * costs[subset] * treeletTriangles);
            partitionMasks[subset] = static_cast<char>(optimalPartition);
        }
    }

    WARP_SYNC;

    // Process subsets of size treeletSize
    float optimalCost = FLT_MAX;
    int optimalPartition = 0;
    int subset = (1 << treeletSize) - 1;
    int partition = (THREAD_WARP_INDEX + 1) * 2;
    int numberOfPartitions = (1 << (treeletSize - 1)) - 1;
    for (int j = THREAD_WARP_INDEX; j < numberOfPartitions; j += 32)
    {
        float partitionCost = costs[partition] + costs[partition ^ subset];
        if (partitionCost < optimalCost)
        {
            optimalCost = partitionCost;
            optimalPartition = partition;
        }

        partition += 64;
    }

    reduceOptimal(optimalCost, optimalPartition, WARP_SIZE);
    if (THREAD_WARP_INDEX == 0)
    {
        // Calculate subset SAH. Keep whichever has a lower SAH between collapsing 
        // the subset treelet or leaving it as is
        costs[subset] = min(ci * costs[subset] + optimalCost,
                ct * costs[subset] * treeletTriangles);
        partitionMasks[subset] = static_cast<char>(optimalPartition);
    }
}

__device__ void updateTreelet(int treeletSize, BVHTree* tree, int* treeletInternalNodes, 
        int* treeletLeaves, float* subsetAreas, float* costs, char* partitionMasks, 
        float* nodesSah, float4* boundingBoxesMin, float4* boundingBoxesMax, int* stackNode, 
        char* stackMask, int* stackSize, int* currentInternalNode)
{
    int globalWarpIndex = GLOBAL_WARP_INDEX;
    if (costs[(1 << treeletSize) - 1] < nodesSah[treeletInternalNodes[0]])
    {
        if (THREAD_WARP_INDEX == 0)
        {
            stackNode[globalWarpIndex * (treeletSize - 1)] = treeletInternalNodes[0];
            stackMask[globalWarpIndex * (treeletSize - 1)] =
                    static_cast<char>((1 << treeletSize) - 1);
            stackSize[globalWarpIndex] = 1;
            currentInternalNode[globalWarpIndex] = 1;
        }

        while (stackSize[globalWarpIndex] > 0)
        {
            int lastStackSize = stackSize[globalWarpIndex];
            if (THREAD_WARP_INDEX == 0)
            {
                stackSize[globalWarpIndex] = 0;
            }
            
            if (THREAD_WARP_INDEX < lastStackSize)
            {
                int nodeSubset = stackMask
                        [globalWarpIndex * (treeletSize - 1) + THREAD_WARP_INDEX];
                char partition = partitionMasks[nodeSubset];
                char partitionComplement = partition ^ nodeSubset;
                int subsetRoot =
                        stackNode[globalWarpIndex * (treeletSize - 1) + THREAD_WARP_INDEX];

                int childIndex;
                if (__popc(partition) > 1)
                {
                    // Update node pointers
                    int currentNode = atomicAdd(currentInternalNode + globalWarpIndex, 1);
                    childIndex =  treeletInternalNodes[currentNode];
                    tree->SetLeftIndex(subsetRoot, childIndex);
                    tree->SetParentIndex(childIndex, subsetRoot);

                    int position = (1 << treeletSize) * globalWarpIndex +
                            partition;
                    float4 bbMin = boundingBoxesMin[position];
                    float4 bbMax = boundingBoxesMax[position];
                    float area = calculateBoundingBoxSurfaceArea(bbMin, bbMax);

                    // Update node area and bounding box                    
                    tree->SetBoundingBoxMin(childIndex, bbMin);
                    tree->SetBoundingBoxMax(childIndex, bbMax);
                    tree->SetArea(childIndex, area);
                    nodesSah[childIndex] = costs[partition];

                    // Add child to stack
                    int stackIndex = atomicAdd(stackSize + globalWarpIndex, 1);
                    stackNode[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                            childIndex;
                    stackMask[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                            partition;
                }
                else
                {
                    childIndex = treeletLeaves[__ffs(partition) - 1];
                    tree->SetLeftIndex(subsetRoot, childIndex);
                    tree->SetParentIndex(childIndex, subsetRoot);
                }
                
                if (__popc(partitionComplement) > 1)
                {
                    // Update node pointers
                    int currentNode = atomicAdd(currentInternalNode + globalWarpIndex, 1);
                    int childIndex = treeletInternalNodes[currentNode];
                    tree->SetRightIndex(subsetRoot, childIndex);
                    tree->SetParentIndex(childIndex, subsetRoot);

                    int position = (1 << treeletSize) * globalWarpIndex +
                            partitionComplement;
                    float4 bbMin = boundingBoxesMin[position];
                    float4 bbMax = boundingBoxesMax[position];
                    float area = calculateBoundingBoxSurfaceArea(bbMin, bbMax);

                    // Update node area and bounding box
                    tree->SetBoundingBoxMin(childIndex, bbMin);
                    tree->SetBoundingBoxMax(childIndex, bbMax);
                    tree->SetArea(childIndex, area);
                    nodesSah[childIndex] = costs[partition];

                    // Add child to stack
                    int stackIndex = atomicAdd(stackSize + globalWarpIndex, 1);
                    stackNode[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                            childIndex;
                    stackMask[globalWarpIndex * (treeletSize - 1) + stackIndex] =
                            partitionComplement;
                }
                else
                {
                    int childIndex = treeletLeaves[__ffs(partitionComplement) - 1];
                    tree->SetRightIndex(subsetRoot, childIndex);
                    tree->SetParentIndex(childIndex, subsetRoot);
                }                
            }
        }
    }
}

__global__ void LAUNCH_BOUNDS(128, 12) 
treeletReestructureKernel(unsigned int numberOfTriangles, BVHTree* tree,
        float* nodesSah, int treeletSize, int* subtreeTriangles, unsigned int* counters, 
        int gamma, int* schedule, int numberOfRounds, float4* boundingBoxesMin, 
        float4* boundingBoxesMax, float* subsetAreas, int* stackNode, char* stackMask,
        int* stackSize, int* currentInternalNode, float ci, float ct)
{
    // Split the pre-allocated shared memory into distinct arrays for our treelet
    extern __shared__ int sharedMemory[];
    __shared__ int* treeletInternalNodes;
    __shared__ int* treeletLeaves;
    __shared__ float* treeletLeavesAreas;
    __shared__ float* costs;
    __shared__ char* partitionMasks;
   
    // Having only the first thread perform this assignments and then
    // synchronizing is actually slower than issuing the assignments on all threads
    int numberOfWarps = blockDim.x / WARP_SIZE;
    if (THREAD_WARP_INDEX == 0)
    {
        treeletInternalNodes = sharedMemory;
        treeletLeaves = treeletInternalNodes + (treeletSize - 1) * numberOfWarps;
        treeletLeavesAreas = (float*)(treeletLeaves + treeletSize * numberOfWarps);
        costs = treeletLeavesAreas + treeletSize * numberOfWarps;
        partitionMasks = (char*)(costs + (1 << treeletSize) * numberOfWarps);
    }
    __syncthreads();

    // If this flag is set, the thread will be excluded from the bottom up traversal, but will 
    // still be available to help form and optimiza treelets
    const int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize leaves
    int currentNodeIndex;
    if (threadIndex < numberOfTriangles)
    {
        int leafIndex = threadIndex + numberOfTriangles - 1;
        float area = tree->Area(leafIndex);
        currentNodeIndex = tree->ParentIndex(leafIndex);
        subtreeTriangles[leafIndex] = 1;
        nodesSah[leafIndex] = ct * area;        
    }
    else
    {
        currentNodeIndex = -1;
    }

    while (__ballot(currentNodeIndex >= 0) != 0)
    {
        // Number of threads who already have processed the current node
        unsigned int counter = 0;

        if (currentNodeIndex >= 0)
        {
            counter = atomicAdd(&counters[currentNodeIndex], 1);

            // Only the last thread to arrive is allowed to process the current node. This ensures
            // that both its children will already have been processed
            if (counter == 0)
            {
                currentNodeIndex = -1;
            }
        }

        // How many triangles can be reached by the subtree with root at the current node
        int triangleCount = 0;
        if (counter != 0)
        {
            // Throughout the code, blocks that have loads separated from stores are so organized 
            // in order to increase ILP (Instruction level parallelism)
            int left = tree->LeftIndex(currentNodeIndex);
            int right = tree->RightIndex(currentNodeIndex);
            float area = tree->Area(currentNodeIndex);
            int trianglesLeft = subtreeTriangles[left];
            float sahLeft = nodesSah[left];
            int trianglesRight = subtreeTriangles[right];            
            float sahRight = nodesSah[right];

            triangleCount = trianglesLeft + trianglesRight;
            subtreeTriangles[currentNodeIndex] = triangleCount;
            nodesSah[currentNodeIndex] = ci * area + sahLeft + sahRight;
        }

        // Check which threads in the warp have treelets to be processed. We are only going to 
        // process a treelet if the current node is the root of a subtree with at least gamma 
        // triangles
        unsigned int vote = __ballot(triangleCount >= gamma);

        while (vote != 0)
        {
            // Get the thread index for the treelet that will be processed            
            int rootThreadIndex = __ffs(vote) - 1;            

            // Get the treelet root by reading the corresponding thread's currentNodeIndex private 
            // variable
            int treeletRootIndex = __shfl(currentNodeIndex, rootThreadIndex);
            
            formTreelet(treeletRootIndex, numberOfTriangles, tree, treeletSize, 
                    WARP_ARRAY(treeletInternalNodes, treeletSize - 1), 
                    WARP_ARRAY(treeletLeaves, treeletSize), 
                    WARP_ARRAY(treeletLeavesAreas, treeletSize));
            
            // Optimize treelet

            calculateSubsetSurfaceAreas(treeletSize, tree, WARP_ARRAY(treeletLeaves, treeletSize), 
                    subsetAreas, boundingBoxesMin, boundingBoxesMax, 
                    WARP_ARRAY(costs, (1 << treeletSize)));
            
            // Set leaves cost
            if (THREAD_WARP_INDEX < treeletSize)
            {
                int leafIndex = WARP_ARRAY_INDEX(1 << THREAD_WARP_INDEX, 1 << treeletSize);
                int treeletLeafIndex = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                costs[leafIndex] = nodesSah[treeletLeaves[treeletLeafIndex]];
            }

            WARP_SYNC;
            
            int treeletTriangles = 
                    subtreeTriangles[treeletInternalNodes[WARP_ARRAY_INDEX(0, treeletSize - 1)]];
            
            // Process subsets of sizes 2 to treeletSize-2 using the schedule
            processSchedule(numberOfRounds, schedule, WARP_ARRAY(costs, (1 << treeletSize)), 
                    WARP_ARRAY(partitionMasks, (1 << treeletSize)), treeletTriangles, ci, ct);

            WARP_SYNC;
            
            // Procecss remaining subsets
            processSubsets(treeletSize, treeletTriangles, WARP_ARRAY(costs, (1 << treeletSize)), 
                    WARP_ARRAY(partitionMasks, (1 << treeletSize)), ci, ct);
            
            WARP_SYNC;
            
            updateTreelet(treeletSize, tree, WARP_ARRAY(treeletInternalNodes, treeletSize - 1), 
                    WARP_ARRAY(treeletLeaves, treeletSize), subsetAreas, 
                    WARP_ARRAY(costs, (1 << treeletSize)), 
                    WARP_ARRAY(partitionMasks, (1 << treeletSize)), nodesSah, boundingBoxesMin, 
                    boundingBoxesMax, stackNode, stackMask, stackSize, currentInternalNode);                    
   
            // Update vote so each treelet is only processed once (set the bit that represents the 
            // treelet that will be processed back to 0)
            vote &= ~(1 << rootThreadIndex);
        }

        // Update current node pointer
        if (currentNodeIndex >= 0)
        {
            currentNodeIndex = tree->ParentIndex(currentNodeIndex);
        }
    }
}

float DeviceTreeletReestructureOptimizer(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, int* subtreeTrianglesCount, float* nodesSah, int treeletSize, 
        int gamma, int* schedule, int numberOfRounds, float4* boundingBoxesMin, 
        float4* boundingBoxesMax, float* subsetAreas, int* stackNode, char* stackMask,
        int* stackSize, int* currentInternalNode, float ci, float ct)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);

    size_t treeletMemorySize = 
            static_cast<size_t>((2 * treeletSize - 1) * sizeof(int) + treeletSize * sizeof(float));
    size_t costAndMaskSize = static_cast<size_t>(
            (1 << treeletSize) * sizeof(float) + (1 << treeletSize) * sizeof(char));
    size_t sharedMemorySize = static_cast<size_t>(
            (treeletMemorySize + costAndMaskSize) * (blockSize.x / 32));

    cudaFuncSetCacheConfig(treeletReestructureKernel, cudaFuncCachePreferShared);

    return TimeKernelExecution([&]()
    {
        treeletReestructureKernel<<<gridSize, blockSize, sharedMemorySize>>>(numberOfTriangles, 
                tree, nodesSah, treeletSize, subtreeTrianglesCount, counters, gamma, schedule,
                numberOfRounds, boundingBoxesMin, boundingBoxesMax, subsetAreas, stackNode, 
                stackMask, stackSize, currentInternalNode, ci, ct);
    });
}

}
