#pragma once

#include <vector_types.h>

#include "BVHTree.h"

namespace BVHRT
{

/// <summary> Bounding Volume Hierarchy is a tree structure commonly used to accelerate ray
///           intersection tests. This particular implementation uses a Structure of Arrays memory
///           layout for its data, which should help increase cache hit rate when accessing only a
///           portion of the information contained in each node.
///
///           <para> Instances of this class must be created using
///                  <see cref="BVHTreeInstanceManager"/>. </para>
///
///           <para> This class can be used in host and device memory spaces. </para> </summary>
///
/// <remarks> Leonardo, 12/16/2014. </remarks>
class SoABVHTree
{
    // Internal data structure
    struct SoABVHData
    {
        int* parentIndices;
        int* leftIndices;
        int* rightIndices;
        int* dataIndices;
        float4* boundingBoxMin;
        float4* boundingBoxMax;
        float* area;
        int rootIndex;
    };

public:

    /// <summary> This destructor can only be used for instances allocated in host memory space. 
    ///           For device space instances, use
    ///           <see cref="BVHTreeInstanceManager::FreeDeviceTree(BVHTree*)"/> instead. 
    ///           </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ~SoABVHTree();

    friend class BVHTreeInstanceManager;

    /// <summary> Gets the tree root node index. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <returns> The tree root node index. </returns>
    __device__ __host__ int RootIndex() const
    {
        return data.rootIndex;
    }

    /// <summary> Sets the tree root node index. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="rootIndex"> The tree root node index. </param>
    __device__ __host__ void SetRootIndex(int rootIndex)
    {
        data.rootIndex = rootIndex;
    }

    /// <summary> Gets the parent index of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    ///
    /// <returns> The parent node,  or -1 if the specified node is the tree root. </returns>
    __device__ __host__ int ParentIndex(int nodeIndex) const
    {
        return data.parentIndices[nodeIndex];
    }

    /// <summary> Sets the parent index of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex">   Node index. </param>
    /// <param name="parentIndex"> Parent index. </param>
    __device__ __host__ void SetParentIndex(int nodeIndex, int parentIndex)
    {
        data.parentIndices[nodeIndex] = parentIndex;
    }

    /// <summary> Gets the left child index of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    ///
    /// <returns> Left child index, or an undetermined value if the node is a leaf. </returns>
    __device__ __host__ int LeftIndex(int nodeIndex) const
    {
        return data.leftIndices[nodeIndex];
    }

    /// <summary> Sets the left child index of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    /// <param name="leftIndex"> Left child index. </param>
    __device__ __host__ void SetLeftIndex(int nodeIndex, int leftIndex)
    {
        data.leftIndices[nodeIndex] = leftIndex;
    }

    /// <summary> Gets the right child index of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    ///
    /// <returns> Right child index, or an undetermined value if the node is a leaf. </returns>
    __device__ __host__ int RightIndex(int nodeIndex) const
    {
        return data.rightIndices[nodeIndex];
    }

    /// <summary> Sets the right child index of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    /// <param name="rightIndex"> Right child index. </param>
    __device__ __host__ void SetRightIndex(int nodeIndex, int rightIndex)
    {
        data.rightIndices[nodeIndex] = rightIndex;
    }

    /// <summary> Gets the data index of the specified node. The returned value represents the 
    ///           index of the triangle that originated this node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Zero-based index of the node. </param>
    ///
    /// <returns> An int. </returns>
    __device__ __host__ int DataIndex(int nodeIndex) const
    {
        return data.dataIndices[nodeIndex];
    }

    /// <summary> Sets the data index of the specified node. Data index is the index of the 
    ///           triangle that originated this node.</summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    /// <param name="dataIndex"> Data index. </param>
    __device__ __host__ void SetDataIndex(int nodeIndex, int dataIndex)
    {
        data.dataIndices[nodeIndex] = dataIndex;
    }

    /// <summary> Gets the minimum coordinates for the bounding box of the specified node. 
    ///           </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    ///
    /// <returns> Bounding box minimum values. The value of the 'w' coordinate is undefined. 
    ///           </returns>
    __device__ __host__ float4 BoundingBoxMin(int nodeIndex) const
    {
        return data.boundingBoxMin[nodeIndex];
    }

    /// <summary> Sets the minimum coordinates for the bounding box of the specified node. 
    ///           </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex">      Node index. </param>
    /// <param name="boundingBoxMin"> Bounding box minimum values. The value of the 'w' coordinate 
    ///                               is ignored. </param>
    __device__ __host__ void SetBoundingBoxMin(int nodeIndex, float4 boundingBoxMin)
    {
        data.boundingBoxMin[nodeIndex] = boundingBoxMin;
    }

    /// <summary> Gets the maximum coordinates for the bounding box of the specified node. 
    ///           </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    ///
    /// <returns> Bounding box maximum values. The value of the 'w' coordinate is undefined. 
    ///           </returns>
    __device__ __host__ float4 BoundingBoxMax(int nodeIndex) const
    {
        return data.boundingBoxMax[nodeIndex];
    }

    /// <summary> Sets the maximum coordinates for the bounding box of the specified node. 
    ///           </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex">      Node index. </param>
    /// <param name="boundingBoxMax"> Bounding box maximum values. The value of the 'w' coordinate 
    ///                               is ignored. </param>
    __device__ __host__ void SetBoundingBoxMax(int nodeIndex, float4 boundingBoxMax)
    {
        data.boundingBoxMax[nodeIndex] = boundingBoxMax;
    }

    /// <summary> Gets the number of triangles contained in the tree. This number is not 
    ///           necessarily the same as the scene's number of triangles, since triangles may 
    ///           have been split. This is also the number of leaves in the tree. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <returns> Number of triangles. </returns>
    __device__ __host__ unsigned int NumberOfTriangles() const
    {
        return numberOfTriangles;
    }

    /// <summary> Gets the surface area of the bounding box of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index </param>
    ///
    /// <returns> Surface area. </returns>
    __device__ __host__ float Area(int nodeIndex) const
    {
        return data.area[nodeIndex];
    }

    /// <summary> Sets the surface area of the bounding box of the specified node. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeIndex"> Node index. </param>
    /// <param name="area">      Surface area. </param>
    __device__ __host__ void SetArea(int nodeIndex, float area)
    {
        data.area[nodeIndex] = area;
    }

    /// <summary> Calculate the Surface Area Heuristic (SAH) for this tree.
    ///
    ///           <para> The default parameter values are set according to those used in "Fast 
    ///                  Parallel Construction of High-Quality Bounding Volume Hierarchies". 
    ///                  </para>
    ///           </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="nodeTraversalCost">        Relative cost of an internal node traversal. 
    ///                                         </param>
    /// <param name="triangleIntersectionCost"> Relative cost of a leaf traversal. </param>
    ///
    /// <returns> SAH value. </returns>
    __host__ float SAH(float nodeTraversalCost = 1.2f, float triangleIntersectionCost = 1.0f) 
            const;

    /// <summary> Dumps the tree data to a file. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="fileLocation"> The file location. </param>
    __host__ void DumpTree(const char* fileLocation) const;

private:

    /// <summary> Constructor. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="numberOfTriangles"> Number of triangles. </param>
    /// <param name="allocateMemory">    true to allocate host memory arrays, false to not 
    ///                                  allocate any array. </param>
    SoABVHTree(unsigned int numberOfTriangles, bool allocateMemory = true);

    SoABVHData data;
    unsigned int numberOfTriangles;
};

}
