#pragma once

#include "BVHCollapsedTree.h"
#include "BVHTree.h"

namespace BVHRT
{

/// <summary> Copy the elements that will not be changed when collapsing the tree from 
///           source to destination. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="source">            Source BVH tree. </param>
/// <param name="destination">       [out] Destination BVH tree. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceCopyTree(unsigned int numberOfTriangles, BVHTree* source,
        BVHCollapsedTree* destination);

/// <summary> Calculate the SAH cost for each tree node. If collapsing all leaves under a node 
///           would result in a lower SAH cost than the original, mark that node to be 
///           collapsed. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="source">            Source BVH tree. </param>
/// <param name="destination">       [out] Destination BVH tree. </param>
/// <param name="counters">          Atomic counters. Each element must be set to 0xFFFFFFFF. 
///                                  </param>
/// <param name="cost">              [out] SAH cost for each internal node. </param>
/// <param name="collapse">          [out] One if the corresponding node should be collapsed, 
///                                  zero otherwise. </param>
/// <param name="ci">                Cost of traversing an internal node. </param>
/// <param name="ct">                Cost of performing a ray-triangle intersection. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceCalculateCosts(unsigned int numberOfTriangles, BVHTree* source,
        BVHCollapsedTree* destination, unsigned int* counters, float* cost, int* collapse, 
        float ci, float ct);
    
/// <summary> Collapse all nodes whose 'collapse' value is one, making those nodes leaves and 
///           adding all descending triangles directly to those nodes. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="source">            Source BVH tree. </param>
/// <param name="destination">       [in,out] Destination BVH tree. </param>
/// <param name="counters">          Atomic counters. Each element must be set to 0xFFFFFFFF. 
///                                  </param>
/// <param name="collapse">          [out] One if the corresponding node should be collapsed, 
///                                  zero otherwise. </param>
/// <param name="dataPosition">      Current position in the triangle array. Must start as 
///                                  zero. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceCollapseTree(unsigned int numberOfTriangles, BVHTree* source, 
        BVHCollapsedTree* destination, unsigned int* counters, int* collapse, 
        int* dataPosition);

}
