#pragma once

#include "BVHTree.h"

namespace BVHRT
{

/// <summary> Optimize a BVH tree using the ATRBVH method. This method consists in assembling
///           treelets for each of the tree nodes and optimizing them using a bottom-up, 
///           agglomerative clustering approach. It is an extension of the method described in 
///           "KARRAS, T., AND AILA, T. 2013. Fast parallel construction of high-quality bounding 
///           volume hierarchies. In Proc. High-Performance Graphics.".
///
///           <para>This method should only be used for treelet sizes of 25 or greater.</para>
///           </summary>
///
/// <remarks> Leonardo, 04/03/2015. </remarks>
///
/// <param name="numberOfTriangles">     Number of triangles. </param>
/// <param name="tree">                  [in,out] The BVH tree. </param>
/// <param name="counters">              Atomic counters. All elements must be set to zero before 
///                                      starting. </param>
/// <param name="subtreeTrianglesCount"> Number of triangles that descend from each internal node. 
///                                      </param>
/// <param name="treeletSize">           The treelet size. </param>
/// <param name="gamma">                 Gamma parameter. For a node to be processed, gamma or 
///                                      more triangles must descend from it. </param>
/// <param name="schedule">              A schedule all the possible node combinations of size 2 
///                                      in a order such that their distances can be stored using 
///                                      coalesced memory accesses. </param>
/// <param name="scheduleSize">          Number of elements in the schedule. </param>
/// <param name="distanceMatrix">        Memory allocated for the distance matrix. </param>
/// <param name="distanceMatrixSize">    Maximum number of elements in the distance matrix </param>
/// <param name="nodesSah">              SAH cost for each tree node. </param>
/// <param name="ci">                    Cost of traversing an internal node. </param>
/// <param name="ct">                    Cost of performing a ray-triangle intersection. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceAgglomerativeTreeletOptimizer(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, int* subtreeTrianglesCount, int treeletSize, int gamma, 
        const int* schedule, int scheduleSize, float* distanceMatrix, int distanceMatrixSize, 
        float* nodesSah, float ci = 1.2f, float ct = 1.0f);

/// <summary> Optimize a BVH tree using the ATRBVH method. This method consists in assembling
///           treelets for each of the tree nodes and optimizing them using a bottom-up, 
///           agglomerative clustering approach. It is an extension of the method described in 
///           "KARRAS, T., AND AILA, T. 2013. Fast parallel construction of high-quality bounding 
///           volume hierarchies. In Proc. High-Performance Graphics.".
///
///           <para>This method should only be used for treelet sizes of 24 or less.</para>
///           </summary>
///
/// <remarks> Leonardo, 04/03/2015. </remarks>
///
/// <param name="numberOfTriangles">     Number of triangles. </param>
/// <param name="tree">                  [in,out] The BVH tree. </param>
/// <param name="counters">              Atomic counters. All elements must be set to zero before 
///                                      starting. </param>
/// <param name="subtreeTrianglesCount"> Number of triangles that descend from each internal node. 
///                                      </param>
/// <param name="treeletSize">           The treelet size. </param>
/// <param name="gamma">                 Gamma parameter. For a node to be processed, gamma or 
///                                      more triangles must descend from it. </param>
/// <param name="schedule">              A schedule all the possible node combinations of size 2 
///                                      in a order such that their distances can be stored using 
///                                      coalesced memory accesses. </param>
/// <param name="scheduleSize">          Number of elements in the schedule. </param>
/// <param name="nodesSah">              SAH cost for each tree node. </param>
/// <param name="ci">                    Cost of traversing an internal node. </param>
/// <param name="ct">                    Cost of performing a ray-triangle intersection. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceAgglomerativeSmallTreeletOptimizer(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, int* subtreeTrianglesCount, int treeletSize, int gamma, 
        const int* schedule, int scheduleSize, float* nodesSah, float ci = 1.2f, float ct = 1.0f);

}
