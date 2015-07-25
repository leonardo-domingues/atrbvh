#include "AgglomerativeTreelet.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include <cfloat>

#include "AgglomerativeScheduler.h"
#include "Commons.cuh"
#include "TimeKernelExecution.h"
#include "Treelet.cuh"
#include "TriangularMatrix.cuh"

namespace BVHRT
{

__device__ void calculateDistancesMatrix(const int* schedule, int scheduleSize, 
        float* distancesMatrix, int distanceMatrixSize, float* bbMin, float* bbMax)
{
    int numberOfIterations = (scheduleSize + (WARP_SIZE - 1)) / WARP_SIZE;
    for (int j = 0; j < numberOfIterations; ++j)
    {
        int element = 0;
        int elementIndex = THREAD_WARP_INDEX + j * WARP_SIZE;
        if (elementIndex < scheduleSize)
        {
            element = schedule[elementIndex];
        }
        int a = element >> 24;
        int b = ((element >> 16) & 0xFF);

        // Read bounding boxes                    
        float3 bbMinA, bbMaxA, bbMinB, bbMaxB;
        SHFL_FLOAT3(bbMinA, bbMin, a);
        SHFL_FLOAT3(bbMaxA, bbMax, a);
        SHFL_FLOAT3(bbMinB, bbMin, b);
        SHFL_FLOAT3(bbMaxB, bbMax, b);

        if (a != 0 || b != 0)
        {
            float distance = calculateBoundingBoxAndSurfaceArea(bbMinA, bbMaxA, bbMinB,
                    bbMaxB);
            distancesMatrix[LOWER_TRM_INDEX(a, b)] = distance;
        }

        a = ((element >> 8) & 0xFF);
        b = (element & 0xFF);

        // Read bounding boxes
        SHFL_FLOAT3(bbMinA, bbMin, a);
        SHFL_FLOAT3(bbMaxA, bbMax, a);
        SHFL_FLOAT3(bbMinB, bbMin, b);
        SHFL_FLOAT3(bbMaxB, bbMax, b);

        if (a != 0 || b != 0)
        {
            float distance = calculateBoundingBoxAndSurfaceArea(bbMinA, bbMaxA, bbMinB,
                    bbMaxB);
            distancesMatrix[LOWER_TRM_INDEX(a, b)] = distance;
        }
    }
}

__device__ void findMinimum(int numberOfElements, int& index, float& distance)
{
    int shflAmount = numberOfElements / 2;
    while (numberOfElements > 1)
    {
        int otherIndex = __shfl_down(index, shflAmount);
        float otherArea = __shfl_down(distance, shflAmount);

        if (otherArea < distance)
        {
            distance = otherArea;
            index = otherIndex;
        }
        numberOfElements = (numberOfElements + 1) / 2;
        shflAmount = numberOfElements / 2;
    }
}

__device__ void findMinimumDistance(float* distancesMatrix, int lastRow, int& minIndex)
{
    float minDistance = FLT_MAX;
    int matrixSize = sumArithmeticSequence(lastRow, 1, lastRow);

    for (int j = THREAD_WARP_INDEX; j < matrixSize; j += WARP_SIZE)
    {
        float distance = distancesMatrix[j];
        if (distance < minDistance)
        {
            minDistance = distance;
            minIndex = j;
        }
    }
    findMinimum(WARP_SIZE, minIndex, minDistance);
    minIndex = __shfl(minIndex, 0);
}

__device__ void updateState(int joinRow, int joinCol, int lastRow, int& threadNode, 
        float& threadSah, int* treeletInternalNodes, int* treeletLeaves, 
        float* treeletLeavesAreas, float* bbMin, float* bbMax, float ci)
{
    // Update 'joinCol' bounding box and update treelet. The left and right indices 
    // and the bounding boxes must be read outside the conditional or else __shfl is 
    // not going to work
    float3 bbMinA, bbMaxA, bbMinB, bbMaxB;
    float sah = __shfl(threadSah, joinRow) + __shfl(threadSah, joinCol);
    int leftIndex = __shfl(threadNode, joinRow);
    int rightIndex = __shfl(threadNode, joinCol);
    SHFL_FLOAT3(bbMinA, bbMin, joinRow);
    SHFL_FLOAT3(bbMaxA, bbMax, joinRow);
    SHFL_FLOAT3(bbMinB, bbMin, joinCol);
    SHFL_FLOAT3(bbMaxB, bbMax, joinCol);
    expandBoundingBox(bbMinA, bbMaxA, bbMinB, bbMaxB);
    if (THREAD_WARP_INDEX == joinCol)
    {
        threadNode = treeletInternalNodes[lastRow - 1];
        floatArrayFromFloat3(bbMinA, bbMin);
        floatArrayFromFloat3(bbMaxA, bbMax);
        float area = calculateBoundingBoxSurfaceArea(bbMinA, bbMaxA);
        treeletLeavesAreas[lastRow] = sah + ci * area;
        threadSah = treeletLeavesAreas[lastRow];
    }

    // Update 'joinRow' node and bounding box. The last block only modified 'joinCol', 
    // which won't conflict with this block, so we can synchronize only once after 
    // both blocks
    int lastIndex = __shfl(threadNode, lastRow);
    float sahLast = __shfl(threadSah, lastRow);
    SHFL_FLOAT3(bbMinB, bbMin, lastRow);
    SHFL_FLOAT3(bbMaxB, bbMax, lastRow);
    if (THREAD_WARP_INDEX == joinRow)
    {
        threadNode = lastIndex;
        threadSah = sahLast;
        floatArrayFromFloat3(bbMinB, bbMin);
        floatArrayFromFloat3(bbMaxB, bbMax);
    }

    // Update lastRow with the information required to update the treelet
    if (THREAD_WARP_INDEX == lastRow)
    {
        threadNode = leftIndex;
        treeletLeaves[lastRow] = rightIndex;
        floatArrayFromFloat3(bbMinA, bbMin);
        floatArrayFromFloat3(bbMaxA, bbMax);
    }
}

__device__ void updateDistancesMatrix(int joinRow, int joinCol, int lastRow, 
        float* distancesMatrix, float* bbMin, float* bbMax)
{
    // Copy last row to 'joinRow' row and columns
    int destinationRow = THREAD_WARP_INDEX;
    int destinationCol = destinationRow;
    if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinRow)
    {
        destinationRow = max(joinRow, destinationRow);
        destinationCol = min(joinRow, destinationCol);
        int indexSource = LOWER_TRM_INDEX(lastRow, THREAD_WARP_INDEX);
        float distance = distancesMatrix[indexSource];
        int indexDestination = LOWER_TRM_INDEX(destinationRow, destinationCol);
        distancesMatrix[indexDestination] = distance;
    }

    // Update row and column 'joinCol'
    destinationRow = THREAD_WARP_INDEX;
    destinationCol = destinationRow;
    if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinCol)
    {
        destinationRow = max(joinCol, destinationRow);
        destinationCol = min(joinCol, destinationCol);
    }
    float3 bbMinA, bbMaxA, bbMinB, bbMaxB;
    SHFL_FLOAT3(bbMinA, bbMin, destinationRow);
    SHFL_FLOAT3(bbMaxA, bbMax, destinationRow);
    SHFL_FLOAT3(bbMinB, bbMin, destinationCol);
    SHFL_FLOAT3(bbMaxB, bbMax, destinationCol);
    float distance = calculateBoundingBoxAndSurfaceArea(bbMinA, bbMaxA, bbMinB,
            bbMaxB);
    if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinCol)
    {
        int indexDestination = LOWER_TRM_INDEX(destinationRow, destinationCol);
        distancesMatrix[indexDestination] = distance;
    }
}

__device__ void updateTreelet(int treeletSize, BVHTree* tree, int threadNode, 
        int* treeletInternalNodes, int* treeletLeaves, float* treeletLeavesAreas, 
        float* nodesSah, float* bbMin, float* bbMax)
{
    if (treeletLeavesAreas[1] < nodesSah[treeletInternalNodes[0]])
    {
        if (THREAD_WARP_INDEX >= 1 && THREAD_WARP_INDEX < treeletSize)
        {
            int nodeIndex = treeletInternalNodes[THREAD_WARP_INDEX - 1];
            tree->SetLeftIndex(nodeIndex, threadNode);
            tree->SetRightIndex(nodeIndex, treeletLeaves[THREAD_WARP_INDEX]);
            tree->SetParentIndex(threadNode, nodeIndex);
            tree->SetParentIndex(treeletLeaves[THREAD_WARP_INDEX], nodeIndex);
            nodesSah[nodeIndex] = treeletLeavesAreas[THREAD_WARP_INDEX];
            float4 bbMin4, bbMax4;
            float4FromFromFloatArray(bbMin, bbMin4);
            float4FromFromFloatArray(bbMax, bbMax4);
            tree->SetBoundingBoxMin(nodeIndex, bbMin4);
            tree->SetBoundingBoxMax(nodeIndex, bbMax4);
            tree->SetArea(nodeIndex, calculateBoundingBoxSurfaceArea(bbMin4, bbMax4));
        }
    }
}

__global__ void agglomerativeTreeletKernel(unsigned int numberOfTriangles, BVHTree* tree, 
        int treeletSize, int* subtreeTriangles, unsigned int* counters, int gamma, 
        const int* schedule, int scheduleSize, float* distancesMatrix, int distanceMatrixSize, 
        float* nodesSah, float ci, float ct)
{
    const int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Split the pre-allocated shared memory into distinc arrays for our treelet
    extern __shared__ int sharedMemory[];
    __shared__ int* treeletInternalNodes;
    __shared__ int* treeletLeaves;
    __shared__ float* treeletLeavesAreas;

    // Initialize shared variables
    if (threadIdx.x == 0)
    {
        int numberOfWarps = blockDim.x / WARP_SIZE;
        treeletInternalNodes = sharedMemory;
        treeletLeaves = treeletInternalNodes + (treeletSize - 1) * numberOfWarps;
        treeletLeavesAreas = (float*)(treeletLeaves + treeletSize * numberOfWarps);
    }
    __syncthreads();
        
    // If this flag is set, the thread will be excluded from the bottom up traversal, but will 
    // still be available to help form and optimize treelets
    int currentNodeIndex;
    if (threadIndex < numberOfTriangles)
    {
        int leafIndex = threadIndex + numberOfTriangles - 1;
        subtreeTriangles[leafIndex] = 1;
        currentNodeIndex = tree->ParentIndex(leafIndex);
        float area = tree->Area(leafIndex);
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
            
            // Load bounding boxes
            float bbMin[3], bbMax[3];
            if (THREAD_WARP_INDEX < treeletSize)
            {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                floatArrayFromFloat4(tree->BoundingBoxMin(treeletLeaves[index]), bbMin);
                floatArrayFromFloat4(tree->BoundingBoxMax(treeletLeaves[index]), bbMax);
            }
            
            calculateDistancesMatrix(schedule, scheduleSize, 
                    distancesMatrix + distanceMatrixSize * GLOBAL_WARP_INDEX, distanceMatrixSize,
                    bbMin, bbMax);            

            int threadNode;
            float threadSah;
            if (THREAD_WARP_INDEX < treeletSize)
            {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                threadNode = treeletLeaves[index];
                threadSah = nodesSah[threadNode];
            }
            
            for (int lastRow = treeletSize - 1; lastRow > 0; --lastRow)
            {
                // Find pair with minimum distance                
                int minIndex = 0;
                findMinimumDistance(distancesMatrix + distanceMatrixSize * GLOBAL_WARP_INDEX, 
                        lastRow, minIndex);

                // Update treelet
                int joinRow = LOWER_TRM_ROW(minIndex);
                int joinCol = LOWER_TRM_COL(minIndex);
                
                // Copy last row to 'joinRow' row and columns
                if (THREAD_WARP_INDEX < lastRow && THREAD_WARP_INDEX != joinRow && lastRow > 1)
                {
                    int destinationRow = max(joinRow, THREAD_WARP_INDEX);
                    int destinationCol = min(joinRow, THREAD_WARP_INDEX);
                    int indexSource = distanceMatrixSize * GLOBAL_WARP_INDEX +
                            LOWER_TRM_INDEX(lastRow, THREAD_WARP_INDEX);
                    float distance = distancesMatrix[indexSource];
                    int indexDestination = distanceMatrixSize * GLOBAL_WARP_INDEX +
                        LOWER_TRM_INDEX(destinationRow, destinationCol);
                    distancesMatrix[indexDestination] = distance;
                }
                
                updateState(joinRow, joinCol, lastRow, threadNode, threadSah,
                        WARP_ARRAY(treeletInternalNodes, treeletSize - 1),
                        WARP_ARRAY(treeletLeaves, treeletSize),
                        WARP_ARRAY(treeletLeavesAreas, treeletSize), bbMin, bbMax, ci);

                // Update row and column 'joinCol'
                if (lastRow > 1)
                {
                    updateDistancesMatrix(joinRow, joinCol, lastRow,
                            distancesMatrix + distanceMatrixSize * GLOBAL_WARP_INDEX, bbMin, 
                            bbMax);
                }
            }

            WARP_SYNC;

            updateTreelet(treeletSize, tree, threadNode,
                WARP_ARRAY(treeletInternalNodes, treeletSize - 1),
                WARP_ARRAY(treeletLeaves, treeletSize),
                WARP_ARRAY(treeletLeavesAreas, treeletSize), nodesSah, bbMin, bbMax);
            
            // Update vote so each treelet is only processed once (set the bit that represents the 
            // treelet that will be processed back to 0)
            vote &= ~(1 << rootThreadIndex);

            WARP_SYNC;
        }
        
        // Update current node pointer
        if (currentNodeIndex >= 0)
        {
            currentNodeIndex = tree->ParentIndex(currentNodeIndex);
        }        
    }
}

__global__ LAUNCH_BOUNDS(128, 12)
void agglomerativeSmallTreeletKernel(unsigned int numberOfTriangles, BVHTree* tree, 
        int treeletSize, int* subtreeTriangles, unsigned int* counters, int gamma, 
        const int* schedule, int scheduleSize, int distanceMatrixSize, float* nodesSah, float ci, 
        float ct)
{
    const int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Split the pre-allocated shared memory into distinc arrays for our treelet
    extern __shared__ int sharedMemory[];
    __shared__ int* treeletInternalNodes;
    __shared__ int* treeletLeaves;
    __shared__ float* treeletLeavesAreas;
    __shared__ float* distancesMatrix;

    // Initialize shared variables
    if (threadIdx.x == 0)
    {
        int numberOfWarps = blockDim.x / WARP_SIZE;
        treeletInternalNodes = sharedMemory;
        treeletLeaves = treeletInternalNodes + (treeletSize - 1) * numberOfWarps;
        treeletLeavesAreas = (float*)(treeletLeaves + treeletSize * numberOfWarps);
        distancesMatrix = treeletLeavesAreas + treeletSize * numberOfWarps;
    }
    __syncthreads();

    // If this flag is set, the thread will be excluded from the bottom up traversal, but will 
    // still be available to help form and optimize treelets
    int currentNodeIndex;
    if (threadIndex < numberOfTriangles)
    {
        int leafIndex = threadIndex + numberOfTriangles - 1;
        subtreeTriangles[leafIndex] = 1;
        currentNodeIndex = tree->ParentIndex(leafIndex);
        float area = tree->Area(leafIndex);
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

            // Load bounding boxes
            float bbMin[3], bbMax[3];
            if (THREAD_WARP_INDEX < treeletSize)
            {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                floatArrayFromFloat4(tree->BoundingBoxMin(treeletLeaves[index]), bbMin);
                floatArrayFromFloat4(tree->BoundingBoxMax(treeletLeaves[index]), bbMax);
            }
                       
            calculateDistancesMatrix(schedule, scheduleSize, 
                    WARP_ARRAY(distancesMatrix, distanceMatrixSize), distanceMatrixSize,
                    bbMin, bbMax);
            
            int threadNode;
            float threadSah;
            if (THREAD_WARP_INDEX < treeletSize)
            {
                int index = WARP_ARRAY_INDEX(THREAD_WARP_INDEX, treeletSize);
                threadNode = treeletLeaves[index];
                threadSah = nodesSah[threadNode];
            }

            for (int lastRow = treeletSize - 1; lastRow > 0; --lastRow)
            {
                // Find pair with minimum distance                
                int minIndex = 0;
                findMinimumDistance(WARP_ARRAY(distancesMatrix, distanceMatrixSize), lastRow, 
                        minIndex);

                // Add modifications to a list
                int joinRow = LOWER_TRM_ROW(minIndex);
                int joinCol = LOWER_TRM_COL(minIndex);                                
                updateState(joinRow, joinCol, lastRow, threadNode, threadSah, 
                        WARP_ARRAY(treeletInternalNodes, treeletSize - 1), 
                        WARP_ARRAY(treeletLeaves, treeletSize), 
                        WARP_ARRAY(treeletLeavesAreas, treeletSize), bbMin, bbMax, ci);
                
                // Update distances matrix
                if (lastRow > 1)
                {
                    updateDistancesMatrix(joinRow, joinCol, lastRow, 
                            WARP_ARRAY(distancesMatrix, distanceMatrixSize), bbMin, bbMax);
                }
            }

            WARP_SYNC;
           
            updateTreelet(treeletSize, tree, threadNode, 
                    WARP_ARRAY(treeletInternalNodes, treeletSize - 1), 
                    WARP_ARRAY(treeletLeaves, treeletSize), 
                    WARP_ARRAY(treeletLeavesAreas, treeletSize), nodesSah, bbMin, bbMax);

            // Update vote so each treelet is only processed once (set the bit that represents the 
            // treelet that will be processed back to 0)
            vote &= ~(1 << rootThreadIndex);
            
            WARP_SYNC;
        }
        
        // Update current node pointer
        if (currentNodeIndex >= 0)
        {
            currentNodeIndex = tree->ParentIndex(currentNodeIndex);
        }
    }
}

float DeviceAgglomerativeTreeletOptimizer(unsigned int numberOfTriangles, BVHTree* tree, 
        unsigned int* counters, int* subtreeTrianglesCount, int treeletSize, int gamma, 
        const int* schedule, int scheduleSize, float* distancesMatrix, int distanceMatrixSize, 
        float* nodesSah, float ci, float ct)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    size_t treeletMemorySize = ((2 * treeletSize - 1) * sizeof(int) + treeletSize * sizeof(float));
    size_t sharedMemorySize = treeletMemorySize * (blockSize.x / 32);

    cudaFuncSetCacheConfig(agglomerativeTreeletKernel, cudaFuncCachePreferShared);

    return TimeKernelExecution([&]()
    {
        agglomerativeTreeletKernel<<<gridSize, blockSize, sharedMemorySize>>>(
                numberOfTriangles, tree, treeletSize, subtreeTrianglesCount, counters, gamma, 
                schedule, scheduleSize, distancesMatrix, distanceMatrixSize, nodesSah, ci, ct);
    });
}

float DeviceAgglomerativeSmallTreeletOptimizer(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, int* subtreeTrianglesCount, int treeletSize, int gamma, 
        const int* schedule, int scheduleSize, float* nodesSah, float ci, float ct)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    size_t treeletMemorySize = ((2 * treeletSize - 1) * sizeof(int) + treeletSize * sizeof(float));
    int distanceMatrixSize = sumArithmeticSequence(treeletSize - 1, 1, treeletSize - 1);
    size_t distancesMatrixMemorySize = distanceMatrixSize * sizeof(float);
    size_t sharedMemorySize = (distancesMatrixMemorySize + treeletMemorySize) * (blockSize.x / 32);

    cudaFuncSetCacheConfig(agglomerativeSmallTreeletKernel, cudaFuncCachePreferShared);

    return TimeKernelExecution([&]()
    {
        agglomerativeSmallTreeletKernel<<<gridSize, blockSize, sharedMemorySize>>>(
                numberOfTriangles, tree, treeletSize, subtreeTrianglesCount, counters, gamma, 
                schedule, scheduleSize, distanceMatrixSize, nodesSah, ci, ct);
    });
}

}
