#include "LBVH.h"

#include <cuda_runtime_api.h>
#include <math_functions.h>
#include <device_launch_parameters.h>

#include "Commons.cuh"
#include "Defines.h"
#include "TimeKernelExecution.h"

namespace BVHRT
{
// Calculate the centroid of the triangle AABB
__device__ float3 getTriangleCentroid(float4 vertex1, float4 vertex2, float4 vertex3)
{
    float3 boundingBoxMin;
    float3 boundingBoxMax;

    boundingBoxMin.x = min(vertex1.x, vertex2.x);
    boundingBoxMin.x = min(boundingBoxMin.x, vertex3.x);
    boundingBoxMax.x = max(vertex1.x, vertex2.x);
    boundingBoxMax.x = max(boundingBoxMax.x, vertex3.x);

    boundingBoxMin.y = min(vertex1.y, vertex2.y);
    boundingBoxMin.y = min(boundingBoxMin.y, vertex3.y);
    boundingBoxMax.y = max(vertex1.y, vertex2.y);
    boundingBoxMax.y = max(boundingBoxMax.y, vertex3.y);

    boundingBoxMin.z = min(vertex1.z, vertex2.z);
    boundingBoxMin.z = min(boundingBoxMin.z, vertex3.z);
    boundingBoxMax.z = max(vertex1.z, vertex2.z);
    boundingBoxMax.z = max(boundingBoxMax.z, vertex3.z);

    float3 centroid;
    centroid.x = (boundingBoxMax.x + boundingBoxMin.x) * 0.5f;
    centroid.y = (boundingBoxMax.y + boundingBoxMin.y) * 0.5f;
    centroid.z = (boundingBoxMax.z + boundingBoxMin.z) * 0.5f;

    return centroid;
}

__device__ float3 getBoundingBoxCentroid(float4 bboxMin, float4 bboxMax)
{
    float3 centroid;

    centroid.x = (bboxMin.x + bboxMax.x) / 2.0f;
    centroid.y = (bboxMin.y + bboxMax.y) / 2.0f;
    centroid.z = (bboxMin.z + bboxMax.z) / 2.0f;

    return centroid;
}

__global__ void generateMortonCodesKernel(unsigned int numberOfTriangles, const Scene* scene,
        unsigned int* mortonCodes, unsigned int* sortIndices)
{
    const int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (globalId >= numberOfTriangles)
    {
        return;
    }

    sortIndices[globalId] = globalId;

    float4 vertex1, vertex2, vertex3;
    loadTriangle(globalId, scene->vertices, &vertex1, &vertex2, &vertex3);
    float3 centroid = getTriangleCentroid(vertex1, vertex2, vertex3);

    float3 normalizedCentroid = normalize(centroid, scene->boundingBoxMin, scene->boundingBoxMax);
    unsigned int mortonCode = calculateMortonCode(normalizedCentroid);

    mortonCodes[globalId] = mortonCode;
}

__global__ void generateMortonCodes64Kernel(unsigned int numberOfTriangles, const Scene* scene,
        unsigned long long int* mortonCodes, unsigned int* sortIndices)
{
    const int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (globalId >= numberOfTriangles)
    {
        return;
    }

    sortIndices[globalId] = globalId;

    float4 vertex1, vertex2, vertex3;
    loadTriangle(globalId, scene->vertices, &vertex1, &vertex2, &vertex3);
    float3 centroid = getTriangleCentroid(vertex1, vertex2, vertex3);

    float3 normalizedCentroid = normalize(centroid, scene->boundingBoxMin, scene->boundingBoxMax);
    unsigned long long int mortonCode = calculateMortonCode64(normalizedCentroid);

    mortonCodes[globalId] = mortonCode;
}

__global__ void generateMortonCodesTSKernel(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax, unsigned int* mortonCodes, 
        unsigned int* routingIndices)
{
    const int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (globalId >= numberOfTriangles)
    {
        return;
    }

    routingIndices[globalId] = globalId;

    // Load scene bounding box
    float3 sceneBboxMin = scene->boundingBoxMin;
    float3 sceneBboxMax = scene->boundingBoxMax;

    // Load triangle bounding box
    float4 bboxMin = boundingBoxesMin[globalId];
    float4 bboxMax = boundingBoxesMax[globalId];

    // Calculate bounding box centroid
    float3 centroid = getBoundingBoxCentroid(bboxMin, bboxMax);

    // Calculate morton code
    float3 normalizedCentroid = normalize(centroid, sceneBboxMin, sceneBboxMax);
    unsigned int mortonCode = calculateMortonCode(normalizedCentroid);

    mortonCodes[globalId] = mortonCode;
}

__global__ void generateMortonCodes64TSKernel(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax, unsigned long long int* mortonCodes, 
        unsigned int* routingIndices)
{
    const int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (globalId >= numberOfTriangles)
    {
        return;
    }

    routingIndices[globalId] = globalId;

    // Load scene bounding box
    float3 sceneBboxMin = scene->boundingBoxMin;
    float3 sceneBboxMax = scene->boundingBoxMax;

    // Load triangle bounding box
    float4 bboxMin = boundingBoxesMin[globalId];
    float4 bboxMax = boundingBoxesMax[globalId];

    // Calculate bounding box centroid
    float3 centroid = getBoundingBoxCentroid(bboxMin, bboxMax);

    // Calculate morton code
    float3 normalizedCentroid = normalize(centroid, sceneBboxMin, sceneBboxMax);
    unsigned long long int mortonCode = calculateMortonCode64(normalizedCentroid);

    mortonCodes[globalId] = mortonCode;
}

__device__ int longestCommonPrefix(unsigned int* sortedKeys, unsigned int numberOfElements,
        int index1, int index2, unsigned int key1)
{
    // No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one 
    // thread per internal node)
    if (index2 < 0 || index2 >= numberOfElements)
    {
        return 0;
    }

    unsigned int key2 = sortedKeys[index2];

    if (key1 == key2)
    {
        return 32 + __clz(index1 ^ index2);
    }

    return __clz(key1 ^ key2);
}

__device__ int longestCommonPrefix(unsigned long long int* sortedKeys, 
        unsigned int numberOfElements, int index1, int index2, unsigned long long int key1)
{
    // No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one 
    // thread per internal node)
    if (index2 < 0 || index2 >= numberOfElements)
    {
        return 0;
    }

    unsigned long long int key2 = sortedKeys[index2];

    if (key1 == key2)
    {
        return 64 + __clzll(index1 ^ index2);
    }

    return __clzll(key1 ^ key2);
}

__device__ int sgn(int number)
{
    return (0 < number) - (0 > number);
}

template <typename T> __global__ void buildTreeKernel(unsigned int numberOfTriangles, 
        T* sortedKeys, unsigned int* sortIndices, BVHTree* tree)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (i >= (numberOfTriangles - 1))
    {
        return;
    }

    const T key1 = sortedKeys[i];

    const int lcp1 = longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + 1, key1);
    const int lcp2 = longestCommonPrefix(sortedKeys, numberOfTriangles, i, i - 1, key1);

    const int direction = sgn((lcp1 - lcp2));

    // Compute upper bound for the length of the range
    const int minLcp = longestCommonPrefix(sortedKeys, numberOfTriangles, i, i - direction, key1);
    int lMax = 128;
    while (longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + lMax * direction, key1) > 
            minLcp)
    {
        lMax *= 4;
    }

    // Find other end using binary search
    int l = 0;
    int t = lMax;
    while (t > 1)
    {
        t = t / 2;
        if (longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + (l + t) * direction, key1) >
                minLcp)
        {
            l += t;
        }
    }
    const int j = i + l * direction;

    // Find the split position using binary search
    const int nodeLcp = longestCommonPrefix(sortedKeys, numberOfTriangles, i, j, key1);
    int s = 0;
    int divisor = 2;
    t = l;
    const int maxDivisor = 1 << (32 - __clz(l));
    while (divisor <= maxDivisor)
    {
        t = (l + divisor - 1) / divisor;
        if (longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + (s + t) * direction, key1) >
            nodeLcp)
        {
            s += t;
        }
        divisor *= 2;
    }
    const int splitPosition = i + s * direction + min(direction, 0);

    int leftIndex;
    int rightIndex;

    // Update left child pointer
    if (min(i, j) == splitPosition)
    {
        // Children is a leaf, add the number of internal nodes to the index
        int leafIndex = splitPosition + (numberOfTriangles - 1);
        leftIndex = leafIndex;

        // Set the leaf data index
        tree->SetDataIndex(leafIndex, sortIndices[splitPosition]);
    }
    else
    {
        leftIndex = splitPosition;
    }

    // Update right child pointer
    if (max(i, j) == (splitPosition + 1))
    {
        // Children is a leaf, add the number of internal nodes to the index
        int leafIndex = splitPosition + 1 + (numberOfTriangles - 1);
        rightIndex = leafIndex;

        // Set the leaf data index
        tree->SetDataIndex(leafIndex, sortIndices[splitPosition + 1]);
    }
    else
    {
        rightIndex = splitPosition + 1;
    }

    // Update children indices
    tree->SetRightIndex(i, rightIndex);
    tree->SetLeftIndex(i, leftIndex);

    // Set parent nodes
    tree->SetParentIndex(leftIndex, i);
    tree->SetParentIndex(rightIndex, i);

    tree->SetDataIndex(i, -1);

    // Set the parent of the root node to -1
    if (i == 0)
    {
        tree->SetParentIndex(0, -1);
        tree->SetRootIndex(0);
    }
}

// This kernel is used when triangle splitting is enabled, so the data indices of newly generated
// triangles can be set correctly
template <typename T> __global__ void buildTreeTSKernel(unsigned int numberOfTriangles, 
        T* sortedKeys, unsigned int* dataIndices, BVHTree* tree, unsigned int* routingIndices)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (i >= (numberOfTriangles - 1))
    {
        return;
    }

    const T key1 = sortedKeys[i];

    const int lcp1 = longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + 1, key1);
    const int lcp2 = longestCommonPrefix(sortedKeys, numberOfTriangles, i, i - 1, key1);

    const int direction = sgn((lcp1 - lcp2));

    // Compute upper bound for the length of the range
    const int minLcp = longestCommonPrefix(sortedKeys, numberOfTriangles, i, i - direction, key1);
    int lMax = 128;
    while (longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + lMax * direction, key1) > 
            minLcp)
    {
        lMax *= 4;
    }

    // Find other end using binary search
    int l = 0;
    int t = lMax;
    while (t > 1)
    {
        t = t / 2;
        if (longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + (l + t) * direction, key1) >
            minLcp)
        {
            l += t;
        }
    }
    const int j = i + l * direction;

    // Find the split position using binary search
    const int nodeLcp = longestCommonPrefix(sortedKeys, numberOfTriangles, i, j, key1);
    int s = 0;
    int divisor = 2;
    t = l;
    const int maxDivisor = 1 << (32 - __clz(l));
    while (divisor <= maxDivisor)
    {
        t = (l + divisor - 1) / divisor;
        if (longestCommonPrefix(sortedKeys, numberOfTriangles, i, i + (s + t) * direction, key1) >
            nodeLcp)
        {
            s += t;
        }
        divisor *= 2;
    }
    const int splitPosition = i + s * direction + min(direction, 0);

    int leftIndex;
    int rightIndex;

    // Update left child pointer
    if (min(i, j) == splitPosition)
    {
        // Children is a leaf, add the number of internal nodes to the index
        int leafIndex = splitPosition + (numberOfTriangles - 1);
        leftIndex = leafIndex;

        // Set the leaf data index
        tree->SetDataIndex(leafIndex, dataIndices[routingIndices[splitPosition]]);
    }
    else
    {
        leftIndex = splitPosition;
    }

    // Update right child pointer
    if (max(i, j) == (splitPosition + 1))
    {
        // Children is a leaf, add the number of internal nodes to the index
        int leafIndex = splitPosition + 1 + (numberOfTriangles - 1);
        rightIndex = leafIndex;

        // Set the leaf data index
        tree->SetDataIndex(leafIndex, dataIndices[routingIndices[splitPosition + 1]]);
    }
    else
    {
        rightIndex = splitPosition + 1;
    }

    // Update children indices
    tree->SetRightIndex(i, rightIndex);
    tree->SetLeftIndex(i, leftIndex);

    // Set parent nodes
    tree->SetParentIndex(leftIndex, i);
    tree->SetParentIndex(rightIndex, i);

    tree->SetDataIndex(i, -1);

    // Set the parent of the root node to -1
    if (i == 0)
    {
        tree->SetParentIndex(0, -1);
        tree->SetRootIndex(0);
    }
}

__device__ __forceinline__ void calculateLeafBoundingBox(float4 vertex1, float4 vertex2, 
        float4 vertex3, float4* bbMin, float4* bbMax)
{
    bbMin->x = min(vertex1.x, vertex2.x);
    bbMin->x = min(bbMin->x, vertex3.x);
    bbMin->y = min(vertex1.y, vertex2.y);
    bbMin->y = min(bbMin->y, vertex3.y);
    bbMin->z = min(vertex1.z, vertex2.z);
    bbMin->z = min(bbMin->z, vertex3.z);

    bbMax->x = max(vertex1.x, vertex2.x);
    bbMax->x = max(bbMax->x, vertex3.x);
    bbMax->y = max(vertex1.y, vertex2.y);
    bbMax->y = max(bbMax->y, vertex3.y);
    bbMax->z = max(vertex1.z, vertex2.z);
    bbMax->z = max(bbMax->z, vertex3.z);
}

__device__ __forceinline__ void calculateNodeBoundingBox(float4* bbMin, float4* bbMax, 
        float4* leftBbMin, float4* leftBbMax, float4* rightBbMin, float4* rightBbMax)
{
    float4 bboxMin;
    bboxMin.x = min(leftBbMin->x, rightBbMin->x);
    bboxMin.y = min(leftBbMin->y, rightBbMin->y);
    bboxMin.z = min(leftBbMin->z, rightBbMin->z);

    float4 bboxMax;
    bboxMax.x = max(leftBbMax->x, rightBbMax->x);
    bboxMax.y = max(leftBbMax->y, rightBbMax->y);
    bboxMax.z = max(leftBbMax->z, rightBbMax->z);

    *bbMin = bboxMin;
    *bbMax = bboxMax;
}

__global__ void calculateNodeBoundingBoxesKernel(unsigned int numberOfTriangles, 
        const Scene* scene, BVHTree* tree, unsigned int* counters)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int firstThreadInBlock = blockIdx.x * blockDim.x;
    const int lastThreadInBlock = firstThreadInBlock + blockDim.x - 1;

    // Initialize cache of bounding boxes in shared memory
    extern __shared__ float4 sharedBbMin[];
    __shared__ float4* sharedBbMax;
    if (threadIdx.x == 0)
    {
        sharedBbMax = sharedBbMin + blockDim.x;
    }
    __syncthreads();

    // Check for valid threads
    if (i >= numberOfTriangles)
    {
        return;
    }

    int index = i + numberOfTriangles - 1;
    int dataIndex = tree->DataIndex(index);

    // Set leaves left and right indices
    tree->SetLeftIndex(index, -1);
    tree->SetRightIndex(index, -1);

    // Calculate leaves bounding box
    float4 vertex1, vertex2, vertex3;
    float4 cacheMin, cacheMax;
    float4* vertices = scene->vertices;

    loadTriangle(dataIndex, vertices, &vertex1, &vertex2, &vertex3);
    calculateLeafBoundingBox(vertex1, vertex2, vertex3, &cacheMin, &cacheMax);
    sharedBbMin[threadIdx.x] = cacheMin;
    sharedBbMax[threadIdx.x] = cacheMax;
    tree->SetBoundingBoxMin(index, cacheMin);
    tree->SetBoundingBoxMax(index, cacheMax);

    __syncthreads();

    // Calculate surface area
    tree->SetArea(index, calculateBoundingBoxSurfaceArea(cacheMin, cacheMax));

    int lastNode = index;
    int current = tree->ParentIndex(index);
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

        float4 childBbMin;
        float4 childBbMax;
        if (childThreadId >= firstThreadInBlock && childThreadId <= lastThreadInBlock)
        {
            // If both child nodes were processed by the same block, we can reuse the values
            // cached in shared memory
            int childThreadIdInBlock = childThreadId - firstThreadInBlock;
            childBbMin = sharedBbMin[childThreadIdInBlock];
            childBbMax = sharedBbMax[childThreadIdInBlock];
        }
        else
        {
            // The children were processed in different blocks, so we have to find out if the one
            // that was not processed by this thread was the left or right one
            int childIndex = tree->LeftIndex(current);
            if (childIndex == lastNode)
            {
                childIndex = tree->RightIndex(current);
            }

            childBbMin = tree->BoundingBoxMin(childIndex);
            childBbMax = tree->BoundingBoxMax(childIndex);
        }

        __syncthreads();

        // Update node bounding box
        calculateNodeBoundingBox(
                &cacheMin, &cacheMax, &cacheMin, &cacheMax, &childBbMin, &childBbMax);
        sharedBbMin[threadIdx.x] = cacheMin;
        sharedBbMax[threadIdx.x] = cacheMax;
        tree->SetBoundingBoxMin(current, cacheMin);
        tree->SetBoundingBoxMax(current, cacheMax);

        __syncthreads();

        // Calculate surface area
        tree->SetArea(current, calculateBoundingBoxSurfaceArea(cacheMin, cacheMax));

        // Update last processed node
        lastNode = current;

        // Update current node pointer
        current = tree->ParentIndex(current);
    }
}

__global__ void calculateNodeBoundingBoxesTSKernel(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, float4* leavesBboxMin, float4* leavesBboxMax, 
        unsigned int* routingIndices)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int firstThreadInBlock = blockIdx.x * blockDim.x;
    const int lastThreadInBlock = firstThreadInBlock + blockDim.x - 1;

    // Initialize cache of bounding boxes in shared memory
    extern __shared__ float4 sharedBbMin[];
    __shared__ float4* sharedBbMax;
    if (threadIdx.x == 0)
    {
        sharedBbMax = sharedBbMin + blockDim.x;
    }

    // Check for valid threads
    if (i >= numberOfTriangles)
    {
        return;
    }

    int index = i + numberOfTriangles - 1;
    int routingIndex = routingIndices[i];

    // Set leaves left and right indices
    tree->SetLeftIndex(index, -1);
    tree->SetRightIndex(index, -1);

    // Write leaves bounding box to the tree and to cache
    float4 cacheMin, cacheMax;
    cacheMin = leavesBboxMin[routingIndex];
    cacheMax = leavesBboxMax[routingIndex];
    sharedBbMin[threadIdx.x] = cacheMin;
    sharedBbMax[threadIdx.x] = cacheMax;
    tree->SetBoundingBoxMin(index, leavesBboxMin[routingIndex]);
    tree->SetBoundingBoxMax(index, leavesBboxMax[routingIndex]);

    // Calculate surface area
    tree->SetArea(index, calculateBoundingBoxSurfaceArea(cacheMin, cacheMax));

    int lastNode = index;
    int current = tree->ParentIndex(index);
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

        float4 childBbMin;
        float4 childBbMax;
        if (childThreadId >= firstThreadInBlock && childThreadId <= lastThreadInBlock)
        {
            // If both child nodes were processed by the same block, we can reuse the values 
            // cached in shared memory
            int childThreadIdInBlock = childThreadId - firstThreadInBlock;
            childBbMin = sharedBbMin[childThreadIdInBlock];
            childBbMax = sharedBbMax[childThreadIdInBlock];
        }
        else
        {
            // The children were processed in different blocks, so we have to find out if the one
            // that was not processed
            // by this thread was the left or right one
            int childIndex = tree->LeftIndex(current);
            if (childIndex == lastNode)
            {
                childIndex = tree->RightIndex(current);
            }

            childBbMin = tree->BoundingBoxMin(childIndex);
            childBbMax = tree->BoundingBoxMax(childIndex);
        }

        // Update node bounding box
        calculateNodeBoundingBox(
        &cacheMin, &cacheMax, &cacheMin, &cacheMax, &childBbMin, &childBbMax);
        sharedBbMin[threadIdx.x] = cacheMin;
        sharedBbMax[threadIdx.x] = cacheMax;
        tree->SetBoundingBoxMin(current, cacheMin);
        tree->SetBoundingBoxMax(current, cacheMax);

        // Calculate surface area
        tree->SetArea(current, calculateBoundingBoxSurfaceArea(cacheMin, cacheMax));

        // Update last processed node
        lastNode = current;

        // Update current node pointer
        current = tree->ParentIndex(current);
    }
}

float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        unsigned int* mortonCodes, unsigned int* sortIndices)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    cudaFuncSetCacheConfig(generateMortonCodesKernel, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        generateMortonCodesKernel<<<gridSize, blockSize>>>(numberOfTriangles, scene, mortonCodes, 
                sortIndices);
    });
}

float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        unsigned long long int* mortonCodes, unsigned int* sortIndices)
{
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    cudaFuncSetCacheConfig(generateMortonCodes64Kernel, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        generateMortonCodes64Kernel<<<gridSize, blockSize>>>(numberOfTriangles, scene, 
                mortonCodes, sortIndices);
    });
}

float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax, unsigned int* mortonCodes, 
        unsigned int* routingIndices)
{
    dim3 blockSize(512, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    cudaFuncSetCacheConfig(generateMortonCodesTSKernel, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        generateMortonCodesTSKernel<<<gridSize, blockSize>>>(numberOfTriangles, scene, 
                boundingBoxesMin, boundingBoxesMax, mortonCodes, routingIndices);
    });
}

float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax, unsigned long long int* mortonCodes, 
        unsigned int* routingIndices)
{
    dim3 blockSize(512, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    cudaFuncSetCacheConfig(generateMortonCodes64TSKernel, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        generateMortonCodes64TSKernel << <gridSize, blockSize >> >(numberOfTriangles, scene,
            boundingBoxesMin, boundingBoxesMax, mortonCodes, routingIndices);
    });
}

float DeviceBuildTree(unsigned int numberOfTriangles, unsigned int* sortedKeys,
        unsigned int* sortIndices, BVHTree* tree)
{
    unsigned int numberOfElements = numberOfTriangles - 1;
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numberOfElements + (blockSize.x - 1)) / blockSize.x, 1, 1);

    cudaFuncSetCacheConfig(buildTreeKernel<unsigned int>, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        buildTreeKernel<<<gridSize, blockSize>>>(numberOfTriangles, sortedKeys, sortIndices, tree);
    });
}

float DeviceBuildTree(unsigned int numberOfTriangles, unsigned long long int* sortedKeys,
        unsigned int* sortIndices, BVHTree* tree)
{
    unsigned int numberOfElements = numberOfTriangles - 1;
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numberOfElements + (blockSize.x - 1)) / blockSize.x, 1, 1);

    cudaFuncSetCacheConfig(buildTreeKernel<unsigned long long int>, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        buildTreeKernel<unsigned long long int><<<gridSize, blockSize>>>(numberOfTriangles, sortedKeys, sortIndices, tree);
    });
}

float DeviceBuildTree(unsigned int numberOfTriangles, unsigned int* sortedKeys,
        unsigned int* dataIndices, BVHTree* tree, unsigned int* routingIndices)
{
    unsigned int numberOfElements = numberOfTriangles - 1;
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numberOfElements + (blockSize.x - 1)) / blockSize.x, 1, 1);

    cudaFuncSetCacheConfig(buildTreeTSKernel<unsigned int>, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        buildTreeTSKernel<unsigned int> << <gridSize, blockSize >> >(numberOfTriangles, sortedKeys, dataIndices,
                tree, routingIndices);
    });
}

float DeviceBuildTree(unsigned int numberOfTriangles, unsigned long long int* sortedKeys,
        unsigned int* dataIndices, BVHTree* tree, unsigned int* routingIndices)
{
    unsigned int numberOfElements = numberOfTriangles - 1;
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((numberOfElements + (blockSize.x - 1)) / blockSize.x, 1, 1);

    cudaFuncSetCacheConfig(buildTreeTSKernel<unsigned long long int>, cudaFuncCachePreferL1);

    return TimeKernelExecution([&]()
    {
        buildTreeTSKernel<unsigned long long int><<<gridSize, blockSize>>>(numberOfTriangles, sortedKeys, dataIndices,
            tree, routingIndices);
    });
}

float DeviceCalculateNodeBoundingBoxes(unsigned int numberOfTriangles, const Scene* scene,
        BVHTree* tree, unsigned int* counters)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    size_t bboxCacheSize = blockSize.x * sizeof(float4) * 2;

    cudaFuncSetCacheConfig(calculateNodeBoundingBoxesKernel, cudaFuncCachePreferShared);

    return TimeKernelExecution([&]()
    {
        calculateNodeBoundingBoxesKernel<<<gridSize, blockSize, bboxCacheSize>>>(
                numberOfTriangles, scene, tree, counters);
    });
}

float DeviceCalculateInternalNodeBoundingBoxes(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, float4* leavesBboxMin, float4* leavesBboxMax, 
        unsigned int* routingIndices)
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numberOfTriangles + (blockSize.x - 1)) / blockSize.x, 1, 1);
    size_t bboxCacheSize = blockSize.x * sizeof(float4) * 2;

    cudaFuncSetCacheConfig(calculateNodeBoundingBoxesTSKernel, cudaFuncCachePreferShared);

    return TimeKernelExecution([&]()
    {
        calculateNodeBoundingBoxesTSKernel<<<gridSize, blockSize, bboxCacheSize>>>(
                numberOfTriangles, tree, counters, leavesBboxMin, leavesBboxMax, routingIndices);
    });
}
}
