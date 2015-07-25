#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cfloat>

#include "BVHTree.h"
#include "Commons.cuh"

namespace BVHRT
{

/// <summary> Finds the index of the element with the largest area using a reduction. The index 
///           and area of each element are stored in a different thread of the same warp, and 
///           accessed using the __shfl() intrinsic. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfElements"> Number of elements that should be checked. </param>
/// <param name="index">            Element index. </param>
/// <param name="area">             Element area. </param>
__forceinline__ __device__ void findLeafToExpand(int numberOfElements, int& index, float& area)
{
    int shflAmount = numberOfElements / 2;
    while (numberOfElements > 1)
    {
        int otherIndex = __shfl_down(index, shflAmount);
        float otherArea = __shfl_down(area, shflAmount);

        if (otherArea > area)
        {
            area = otherArea;
            index = otherIndex;
        }
        numberOfElements = (numberOfElements + 1) / 2;
        shflAmount = numberOfElements / 2;
    }
}

/// <summary> Assembles a treelet from the specified treelet root node, using the algorithm 
///           described in "KARRAS, T., AND AILA, T. 2013. Fast parallel construction of 
///           high-quality bounding volume hierarchies. In Proc. High-Performance Graphics."
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="treeletRootIndex">     Treelet root index. </param>
/// <param name="numberOfTriangles">    Number of triangles. </param>
/// <param name="tree">                 The BVH tree. </param>
/// <param name="treeletSize">          The treelet size. </param>
/// <param name="treeletInternalNodes"> [out] Array where the treelet internal nodes will be 
///                                     stored. </param>
/// <param name="treeletLeaves">        [out] Array where the treelet leaves will be stored. 
///                                     </param>
/// <param name="treeletLeavesAreas">   [out] Array where the treelet leaves areas will be stored. 
///                                      </param>
__forceinline__ __device__ void formTreelet(int treeletRootIndex, unsigned int numberOfTriangles, 
        BVHTree* tree, int treeletSize, int* treeletInternalNodes, int* treeletLeaves,
        float* treeletLeavesAreas)
{
    // Initialize treelet
    int left = tree->LeftIndex(treeletRootIndex);
    int right = tree->RightIndex(treeletRootIndex);
    float areaLeft = tree->Area(left);
    float areaRight = tree->Area(right);
    if (THREAD_WARP_INDEX == 0)
    {
        treeletInternalNodes[0] = treeletRootIndex;
        treeletLeaves[0] = left;
        treeletLeaves[1] = right;
        treeletLeavesAreas[0] = areaLeft;
        treeletLeavesAreas[1] = areaRight;
    }

    WARP_SYNC;

    // Find the treelet's internal nodes. On each iteration we choose the leaf with 
    // largest surface area and add move it to the list of internal nodes, adding its
    // two children as leaves in its place
    for (int iteration = 0; iteration < treeletSize - 2; ++iteration)
    {
        // Choose leaf with the largest area
        int largestLeafIndex;
        float largestLeafArea;
        if (THREAD_WARP_INDEX < 2 + iteration)
        {
            largestLeafIndex = THREAD_WARP_INDEX;
            largestLeafArea = treeletLeavesAreas[THREAD_WARP_INDEX];
            if (isLeaf(treeletLeaves[largestLeafIndex], numberOfTriangles))
            {
                largestLeafArea = -FLT_MAX;
            }
        }
        findLeafToExpand(2 + iteration, largestLeafIndex, largestLeafArea);

        // Update treelet
        if (THREAD_WARP_INDEX == 0)
        {
            int replace = treeletLeaves[largestLeafIndex];
            int left = tree->LeftIndex(replace);
            int right = tree->RightIndex(replace);
            float areaLeft = tree->Area(left);
            float areaRight = tree->Area(right);

            treeletInternalNodes[iteration + 1] = replace;
            treeletLeaves[largestLeafIndex] = left;
            treeletLeaves[iteration + 2] = right;
            treeletLeavesAreas[largestLeafIndex] = areaLeft;
            treeletLeavesAreas[iteration + 2] = areaRight;
        }

        WARP_SYNC;
    }
}

}
