#pragma once

#include <vector_types.h>

// Uncomment this to enforce warp synchronization
#define SAFE_WARP_SYNCHRONY

// Synchronize warp. This protects the code from future compiler optimization that 
// involves instructions reordering, possibly leading to race conditions. 
// __syncthreads() could be used instead, at a slight performance penalty
#ifdef SAFE_WARP_SYNCHRONY
#define WARP_SYNC \
do { \
    int _sync = 0; \
    __shfl(_sync, 0); \
} while (0)
#else
#define WARP_SYNC \
do { \
} while (0)
#endif;

#ifndef __CUDA_ARCH__
#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef __CUDA_ARCH__
#define LAUNCH_BOUNDS(x, y) __launch_bounds__((x), (y))
#else
#define LAUNCH_BOUNDS(x, y)
#endif

#define WARP_SIZE 32

// Get the global warp index
#define GLOBAL_WARP_INDEX static_cast<int>((threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE)

// Get the block warp index
#define WARP_INDEX static_cast<int>(threadIdx.x / WARP_SIZE)

// Get a pointer to the beginning of a warp area in an array that stores a certain number of 
// elements for each warp
#define WARP_ARRAY(source, elementsPerWarp) ((source) + WARP_INDEX * (elementsPerWarp))

// Calculate the index of a value in an array that stores a certain number of elements for each 
// warp
#define WARP_ARRAY_INDEX(index, elementsPerWarp) (WARP_INDEX * (elementsPerWarp) + (index))

// Index of the thread in the warp, from 0 to WARP_SIZE-1
#define THREAD_WARP_INDEX (threadIdx.x & (WARP_SIZE - 1))

// Read a vector of 3 elements using shuffle operations
#define SHFL_FLOAT3(destination, source, index) \
do { \
    (destination).x = __shfl((source)[0], (index)); \
    (destination).y = __shfl((source)[1], (index)); \
    (destination).z = __shfl((source)[2], (index)); \
} while (0);

namespace BVHRT
{

/// <summary> Checks if the specified index corresponds to an internal node. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="index">             Node index. </param>
/// <param name="numberOfTriangles"> Number of triangles contained in the BVH. </param>
///
/// <returns> true if the index corresponds to an internal node, false otherwise. </returns>
__forceinline__ __host__ __device__ bool isInternalNode(unsigned int index,
        unsigned int numberOfTriangles)
{
    return (index < numberOfTriangles - 1);
}

/// <summary> Checks if the specified index corresponds to a leaf node. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="index">             Node index. </param>
/// <param name="numberOfTriangles"> Number of triangles contained in the BVH. </param>
///
/// <returns> true if the index corresponds to a leaf node, false otherwise. </returns>
__forceinline__ __host__ __device__ bool isLeaf(unsigned int index, unsigned int numberOfTriangles)
{
    return !isInternalNode(index, numberOfTriangles);
}

/// <summary> Calculates the surface area of a bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin"> The bounding box minimum. </param>
/// <param name="bbMax"> The bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__ __host__ __device__ float calculateBoundingBoxSurfaceArea(float3 bbMin,
        float3 bbMax)
{
    float3 size;
    size.x = bbMax.x - bbMin.x;
    size.y = bbMax.y - bbMin.y;
    size.z = bbMax.z - bbMin.z;
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Calculates the surface area of a bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin"> The bounding box minimum. </param>
/// <param name="bbMax"> The bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__ __host__ __device__ float calculateBoundingBoxSurfaceArea(float4 bbMin,
        float4 bbMax)
{
    float3 size;
    size.x = bbMax.x - bbMin.x;
    size.y = bbMax.y - bbMin.y;
    size.z = bbMax.z - bbMin.z;
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Calculates the bounding box surface area. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin"> The bounding box minimum. </param>
/// <param name="bbMax"> The bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__ __host__ __device__ float calculateBoundingBoxSurfaceArea(const float* bbMin,
        const float* bbMax)
{
    float3 size;
    size.x = bbMax[0] - bbMin[0];
    size.y = bbMax[1] - bbMin[1];
    size.z = bbMax[2] - bbMin[2];
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Calculates the union of two bounding boxes and returns the union box surface area. 
///           </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin1"> First bounding box minimum. </param>
/// <param name="bbMax1"> First bounding box maximum. </param>
/// <param name="bbMin2"> Second bounding box minimum. </param>
/// <param name="bbMax2"> Second bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__ __host__ __device__ float calculateBoundingBoxAndSurfaceArea(const float4 bbMin1,
        const float4 bbMax1, const float4 bbMin2, const float4 bbMax2)
{
    float3 size;
    size.x = max(bbMax1.x, bbMax2.x) - min(bbMin1.x, bbMin2.x);
    size.y = max(bbMax1.y, bbMax2.y) - min(bbMin1.y, bbMin2.y);
    size.z = max(bbMax1.z, bbMax2.z) - min(bbMin1.z, bbMin2.z);
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Calculates the union of two bounding boxes and returns the union box surface area. 
///           </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="bbMin1"> First bounding box minimum. </param>
/// <param name="bbMax1"> First bounding box maximum. </param>
/// <param name="bbMin2"> Second bounding box minimum. </param>
/// <param name="bbMax2"> Second bounding box maximum. </param>
///
/// <returns> The calculated bounding box surface area. </returns>
__forceinline__ __host__ __device__ float calculateBoundingBoxAndSurfaceArea(const float3 bbMin1,
    const float3 bbMax1, const float3 bbMin2, const float3 bbMax2)
{
    float3 size;
    size.x = max(bbMax1.x, bbMax2.x) - min(bbMin1.x, bbMin2.x);
    size.y = max(bbMax1.y, bbMax2.y) - min(bbMin1.y, bbMin2.y);
    size.z = max(bbMax1.z, bbMax2.z) - min(bbMin1.z, bbMin2.z);
    return 2 * (size.x * size.y + size.x * size.z + size.y * size.z);
}

/// <summary> Loads a triangle from the vertices array. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="index">    The node index. </param>
/// <param name="vertices"> The array of vertices. </param>
/// <param name="vertex1">  [out] The first vertex. </param>
/// <param name="vertex2">  [out] The second vertex. </param>
/// <param name="vertex3">  [out] The third vertex. </param>
__forceinline__ __host__ __device__ void loadTriangle(int index, const float4* vertices, 
        float4* vertex1, float4* vertex2, float4* vertex3)
{
    *vertex1 = vertices[index * 3];
    *vertex2 = vertices[index * 3 + 1];
    *vertex3 = vertices[index * 3 + 2];
}

/// <summary> Calculates the triangle bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="vertex1">        The first vertex. </param>
/// <param name="vertex2">        The second vertex. </param>
/// <param name="vertex3">        The third vertex. </param>
/// <param name="boundingBoxMin"> [out] The bounding box minimum. </param>
/// <param name="boundingBoxMax"> [out] The bounding box maximum. </param>
__forceinline__ __host__ __device__ void calculateTriangleBoundingBox(float4 vertex1,
        float4 vertex2, float4 vertex3, float3* boundingBoxMin, float3* boundingBoxMax)
{
    boundingBoxMin->x = min(vertex1.x, vertex2.x);
    boundingBoxMin->x = min(boundingBoxMin->x, vertex3.x);
    boundingBoxMax->x = max(vertex1.x, vertex2.x);
    boundingBoxMax->x = max(boundingBoxMax->x, vertex3.x);

    boundingBoxMin->y = min(vertex1.y, vertex2.y);
    boundingBoxMin->y = min(boundingBoxMin->y, vertex3.y);
    boundingBoxMax->y = max(vertex1.y, vertex2.y);
    boundingBoxMax->y = max(boundingBoxMax->y, vertex3.y);

    boundingBoxMin->z = min(vertex1.z, vertex2.z);
    boundingBoxMin->z = min(boundingBoxMin->z, vertex3.z);
    boundingBoxMax->z = max(vertex1.z, vertex2.z);
    boundingBoxMax->z = max(boundingBoxMax->z, vertex3.z);
}

// --- Vector operations --------------------------------------------------------------------------

/// <summary> Gets the coordinate from the specified vector type using its index. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> Source vector. </param>
/// <param name="index">  Coordinate index. </param>
///
/// <returns> The coordinate value. </returns>
__forceinline__ __host__ __device__ float getCoordinate(float3 source, int index)
{
    if (index == 0)
    {
        return source.x;
    }
    else if (index == 1)
    {
        return source.y;
    }
    else
    {
        return source.z;
    }
}

/// <summary> Gets the coordinate from the specified vector type using its index. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> Source vector. </param>
/// <param name="index">  Coordinate index. </param>
///
/// <returns> The coordinate value. </returns>
__forceinline__ __host__ __device__ float getCoordinate(float4 source, int index)
{
    if (index == 0)
    {
        return source.x;
    }
    else if (index == 1)
    {
        return source.y;
    }
    else
    {
        return source.z;
    }
}

/// <summary> Sets the coordinate from the specified vector type using its index. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> [out] Source vector. </param>
/// <param name="index">  Coordinate index. </param>
/// <param name="value">  Value. </param>
__forceinline__ __host__ __device__ void setCoordinate(float4* source, int index, float value)
{
    if (index == 0)
    {
        source->x = value;
    }
    else if (index == 1)
    {
        source->y = value;
    }
    else
    {
        source->z = value;
    }
}

/// <summary> Calculates v1 - v2. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="v1"> v1. </param>
/// <param name="v2"> v2. </param>
///
/// <returns> v1 - v2. </returns>
__forceinline__ __host__ __device__ float3 subtract(float4 v1, float4 v2)
{
    float3 result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    result.z = v1.z - v2.z;
    return result;
}

/// <summary> Calculates the cross product between v1 and v2. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="v1"> v1. </param>
/// <param name="v2"> v2. </param>
///
/// <returns> v1 x v2. </returns>
__forceinline__ __host__ __device__ float3 cross(float3 v1, float3 v2)
{
    float3 result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return result;
}

/// <summary> Converts a float3 to float4. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> The source vector. </param>
///
/// <returns> A float4. </returns>
__forceinline__ __host__ __device__ float4 float4FromFloat3(float3 source)
{
    float4 temp;
    temp.x = source.x;
    temp.y = source.y;
    temp.z = source.z;
    // we do not care about w

    return temp;
}

/// <summary> Converts a float4 to float3. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="source"> The source vector. </param>
///
/// <returns> A float3. </returns>
__forceinline__ __host__ __device__ float3 float3FromFloat4(float4 source)
{
    float3 temp;
    temp.x = source.x;
    temp.y = source.y;
    temp.z = source.z;
    // we do not care about w

    return temp;
}

/// <summary> Converts a float4 to an array of floats. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 01/21/2015. </remarks>
///
/// <param name="source"> The source vector. </param>
/// <param name="destination"> The destination array. </param>
__forceinline__ __host__ __device__ void floatArrayFromFloat4(float4 source, float* destination)
{
    destination[0] = source.x;
    destination[1] = source.y;
    destination[2] = source.z;
}

/// <summary> Converts a float3 to an array of floats. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 01/22/2015. </remarks>
///
/// <param name="source"> The source vector. </param>
/// <param name="destination"> The destination array. </param>
__forceinline__ __host__ __device__ void floatArrayFromFloat3(float3 source, float* destination)
{
    destination[0] = source.x;
    destination[1] = source.y;
    destination[2] = source.z;
}

/// <summary> Converts an array of floats to a float4. The 'w' coordinate is ignored. </summary>
///
/// <remarks> Leonardo, 01/21/2015. </remarks>
///
/// <param name="source"> The source array. </param>
/// <param name="destination"> The destination vector. </param>
__forceinline__ __host__ __device__ void float4FromFromFloatArray(const float* source, 
        float4& destination)
{
    destination.x = source[0];
    destination.y = source[1];
    destination.z = source[2];
}

// --- Morton codes -------------------------------------------------------------------------------

/// <summary> Normalizes a position using the specified bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point">          The point. </param>
/// <param name="boundingBoxMin"> The bounding box minimum. </param>
/// <param name="boundingBoxMax"> The bounding box maximum. </param>
///
/// <returns> Normalized position. </returns>
__forceinline__ __host__ __device__ float3 normalize(float3 point, float3 boundingBoxMin,
        float3 boundingBoxMax)
{
    float3 normalized;
    normalized.x = (point.x - boundingBoxMin.x) / (boundingBoxMax.x - boundingBoxMin.x);
    normalized.y = (point.y - boundingBoxMin.y) / (boundingBoxMax.y - boundingBoxMin.y);
    normalized.z = (point.z - boundingBoxMin.z) / (boundingBoxMax.z - boundingBoxMin.z);
    return normalized;
}

/// <summary> Normalizes a position using the specified bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point">          The point. </param>
/// <param name="boundingBoxMin"> The bounding box minimum. </param>
/// <param name="boundingBoxMax"> The bounding box maximum. </param>
///
/// <returns> Normalized position. </returns>
__forceinline__ __host__ __device__ float4 normalize(float4 point, float4 boundingBoxMin,
        float4 boundingBoxMax)
{
    float4 normalized;
    normalized.x = (point.x - boundingBoxMin.x) / (boundingBoxMax.x - boundingBoxMin.x);
    normalized.y = (point.y - boundingBoxMin.y) / (boundingBoxMax.y - boundingBoxMin.y);
    normalized.z = (point.z - boundingBoxMin.z) / (boundingBoxMax.z - boundingBoxMin.z);
    return normalized;
}

/// <summary> Un-normalizes a position using the specified bounding box. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="normalized"> The normalized value. </param>
/// <param name="bboxMin">    The bounding box minimum. </param>
/// <param name="bboxMax">    The bounding box maximum. </param>
///
/// <returns> The un-normalized value. </returns>
__forceinline__ __host__ __device__ float4 denormalize(float4 normalized, float3 bboxMin,
        float3 bboxMax)
{
    float4 point;
    point.x = bboxMin.x + (bboxMax.x - bboxMin.x) * normalized.x;
    point.y = bboxMin.y + (bboxMax.y - bboxMin.y) * normalized.y;
    point.z = bboxMin.z + (bboxMax.z - bboxMin.z) * normalized.z;
    return point;
}

/// <summary> Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The expanded value. </returns>
__forceinline__ __host__ __device__ unsigned int expandBits(unsigned int value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

/// <summary> Calculates the point morton code using 30 bits. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point"> The point. </param>
///
/// <returns> The calculated morton code. </returns>
__forceinline__ __host__ __device__ unsigned int calculateMortonCode(float3 point)
{
    // Discretize the unit cube into a 10 bit integer
    uint3 discretized;
    discretized.x = (unsigned int)min(max(point.x * 1024.0f, 0.0f), 1023.0f);
    discretized.y = (unsigned int)min(max(point.y * 1024.0f, 0.0f), 1023.0f);
    discretized.z = (unsigned int)min(max(point.z * 1024.0f, 0.0f), 1023.0f);

    discretized.x = expandBits(discretized.x);
    discretized.y = expandBits(discretized.y);
    discretized.z = expandBits(discretized.z);

    return discretized.x * 4 + discretized.y * 2 + discretized.z;
}

/// <summary> Calculates the point morton code using 30 bits. The 'w' coordinate is ignored.
///           </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point"> The point. </param>
///
/// <returns> The calculated morton code. </returns>
__forceinline__ __host__ __device__ unsigned int calculateMortonCode(float4 point)
{
    // Discretize the unit cube into a 10 bit integer
    uint3 discretized;
    discretized.x = (unsigned int)min(max(point.x * 1024.0f, 0.0f), 1023.0f);
    discretized.y = (unsigned int)min(max(point.y * 1024.0f, 0.0f), 1023.0f);
    discretized.z = (unsigned int)min(max(point.z * 1024.0f, 0.0f), 1023.0f);

    discretized.x = expandBits(discretized.x);
    discretized.y = expandBits(discretized.y);
    discretized.z = expandBits(discretized.z);

    return discretized.x * 4 + discretized.y * 2 + discretized.z;
}

/// <summary> Compact bits from the specified 30-bit value, using only one bit at every 3 from the
///           original value and forming a 10-bit value. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The compacted value. </returns>
__forceinline__ __host__ __device__ unsigned int compactBits(unsigned int value)
{
    unsigned int compacted = value;
    compacted &= 0x09249249;
    compacted = (compacted ^ (compacted >> 2)) & 0x030c30c3;
    compacted = (compacted ^ (compacted >> 4)) & 0x0300f00f;
    compacted = (compacted ^ (compacted >> 8)) & 0xff0000ff;
    compacted = (compacted ^ (compacted >> 16)) & 0x000003ff;
    return compacted;
}

/// <summary> Decodes the 'x' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__ __host__ __device__ float decodeMortonCodeX(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 2);

    return expanded / 1024.0f;
}

/// <summary> Decodes the 'y' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__ __host__ __device__ float decodeMortonCodeY(unsigned int value)
{
    unsigned int expanded = compactBits(value >> 1);

    return expanded / 1024.0f;
}

/// <summary> Decodes the 'z' coordinate from a 30-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__ __host__ __device__ float decodeMortonCodeZ(unsigned int value)
{
    unsigned int expanded = compactBits(value);

    return expanded / 1024.0f;
}

/// <summary> Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The expanded value. </returns>
__forceinline__ __host__ __device__ unsigned long long int expandBits64(
        unsigned long long int value)
{
    unsigned long long int expanded = value;
    expanded &= 0x1fffff;
    expanded = (expanded | expanded << 32) & 0x1f00000000ffff;
    expanded = (expanded | expanded << 16) & 0x1f0000ff0000ff;
    expanded = (expanded | expanded << 8) & 0x100f00f00f00f00f;
    expanded = (expanded | expanded << 4) & 0x10c30c30c30c30c3;
    expanded = (expanded | expanded << 2) & 0x1249249249249249;

    return expanded;
}

/// <summary> Calculates the point morton code using 63 bits. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point"> The point. </param>
///
/// <returns> The 63-bit morton code. </returns>
__forceinline__ __host__ __device__ unsigned long long int calculateMortonCode64(float3 point)
{
    // Discretize the unit cube into a 10 bit integer
    unsigned long long int discretized[3];
    discretized[0] = (unsigned long long int)min(max(point.x * 2097152.0f, 0.0f), 2097151.0f);
    discretized[1] = (unsigned long long int)min(max(point.y * 2097152.0f, 0.0f), 2097151.0f);
    discretized[2] = (unsigned long long int)min(max(point.z * 2097152.0f, 0.0f), 2097151.0f);

    discretized[0] = expandBits64(discretized[0]);
    discretized[1] = expandBits64(discretized[1]);
    discretized[2] = expandBits64(discretized[2]);

    return discretized[0] * 4 + discretized[1] * 2 + discretized[2];
}

/// <summary> Calculates the point morton code using 63 bits. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="point"> The point. </param>
///
/// <returns> The 63-bit morton code. </returns>
__forceinline__ __host__ __device__ unsigned long long int calculateMortonCode64(float4 point)
{
    // Discretize the unit cube into a 10 bit integer
    unsigned long long int discretized[3];
    discretized[0] = (unsigned long long int)min(max(point.x * 2097152.0f, 0.0f), 2097151.0f);
    discretized[1] = (unsigned long long int)min(max(point.y * 2097152.0f, 0.0f), 2097151.0f);
    discretized[2] = (unsigned long long int)min(max(point.z * 2097152.0f, 0.0f), 2097151.0f);

    discretized[0] = expandBits64(discretized[0]);
    discretized[1] = expandBits64(discretized[1]);
    discretized[2] = expandBits64(discretized[2]);

    return discretized[0] * 4 + discretized[1] * 2 + discretized[2];
}

/// <summary> Compact bits from the specified 63-bit value, using only one bit at every 3 from the
///           original value and forming a 21-bit value. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The compacted value. </returns>
__forceinline__ __host__ __device__ unsigned long long int compactBits64(
        unsigned long long int value)
{
    unsigned long long int compacted = value;

    compacted &= 0x1249249249249249;
    compacted = (compacted | compacted >> 2) & 0x10c30c30c30c30c3;
    compacted = (compacted | compacted >> 4) & 0x100f00f00f00f00f;
    compacted = (compacted | compacted >> 8) & 0x1f0000ff0000ff;
    compacted = (compacted | compacted >> 16) & 0x1f00000000ffff;
    compacted = (compacted | compacted >> 32) & 0x1fffff;

    return compacted;
}

/// <summary> Decodes the 'x' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__ __host__ __device__ float decodeMortonCode64X(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 2);

    return expanded / 2097152.0f;
}

/// <summary> Decodes the 'y' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__ __host__ __device__ float decodeMortonCode64Y(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value >> 1);

    return expanded / 2097152.0f;
}

/// <summary> Decodes the 'z' coordinate from a 63-bit morton code. The returned value is a float
///           between 0 and 1. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="value"> The value. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__ __host__ __device__ float decodeMortonCode64Z(unsigned long long int value)
{
    unsigned long long int expanded = compactBits64(value);

    return expanded / 2097152.0f;
}

/// <summary> Expands the group bounding box using the specified new bounding box coordinates. 
///           </summary>
///
/// <remarks> Leonardo, 01/22/2015. </remarks>
///
/// <param name="groupBbMin"> Group bounding box minimum values. </param>
/// <param name="groupBbMax"> Group bounding box maximum values. </param>
/// <param name="newBbMin"> New bounding box minimum values. </param>
/// <param name="newBbMax"> New bounding box maximum values. </param>
__forceinline__ __host__ __device__ void expandBoundingBox(float4& groupBbMin, float4& groupBbMax, 
        const float4& newBbMin, const float4& newBbMax)
{
    groupBbMin.x = min(newBbMin.x, groupBbMin.x);
    groupBbMin.y = min(newBbMin.y, groupBbMin.y);
    groupBbMin.z = min(newBbMin.z, groupBbMin.z);

    groupBbMax.x = max(newBbMax.x, groupBbMax.x);
    groupBbMax.y = max(newBbMax.y, groupBbMax.y);
    groupBbMax.z = max(newBbMax.z, groupBbMax.z);
}

/// <summary> Expands the group bounding box using the specified new bounding box coordinates. 
///           </summary>
///
/// <remarks> Leonardo, 01/22/2015. </remarks>
///
/// <param name="groupBbMin"> Group bounding box minimum values. </param>
/// <param name="groupBbMax"> Group bounding box maximum values. </param>
/// <param name="newBbMin"> New bounding box minimum values. </param>
/// <param name="newBbMax"> New bounding box maximum values. </param>
__forceinline__ __host__ __device__ void expandBoundingBox(float3& groupBbMin, float3& groupBbMax,
    const float3& newBbMin, const float3& newBbMax)
{
    groupBbMin.x = min(newBbMin.x, groupBbMin.x);
    groupBbMin.y = min(newBbMin.y, groupBbMin.y);
    groupBbMin.z = min(newBbMin.z, groupBbMin.z);

    groupBbMax.x = max(newBbMax.x, groupBbMax.x);
    groupBbMax.y = max(newBbMax.y, groupBbMax.y);
    groupBbMax.z = max(newBbMax.z, groupBbMax.z);
}

// --- Arithmetic sequence operations -------------------------------------------------------------

/// <summary> Calculates the sum of an arithmetic sequence. </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
///
/// <param name="numberOfElements"> Number of elements in the sequence. </param>
/// <param name="firstElement"> First element in the sequence. </param>
/// <param name="lastElement"> Last element in the sequence. </param>
///
/// <returns> The decoded value. </returns>
__forceinline__ __host__ __device__ int sumArithmeticSequence(int numberOfElements, 
        int firstElement, int lastElement)
{
    return numberOfElements * (firstElement + lastElement) / 2;
}

}
