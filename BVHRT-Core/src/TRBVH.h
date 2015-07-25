#pragma once

#include "BVHTree.h"

namespace BVHRT
{

/// <summary> Optimize a BVH tree using the TRBVH method, described in detail in "KARRAS, T., 
///           AND AILA, T. 2013. Fast parallel construction of high-quality bounding volume 
///           hierarchies. In Proc. High-Performance Graphics.". </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles">     Number of triangles. </param>
/// <param name="tree">                  [in,out] The BVH tree. </param>
/// <param name="counters">              Atomic counters. All elements must be set to zero before 
///                                      starting. </param>
/// <param name="subtreeTrianglesCount"> Number of triangles that descend from each internal node. 
///                                      </param>
/// <param name="nodesSah">              SAH cost for each tree node. </param>
/// <param name="treeletSize">           The treelet size. </param>
/// <param name="gamma">                 Gamma parameter. For a node to be processed, gamma or 
///                                      more triangles must descend from it. </param>
/// <param name="schedule">              A schedule indicating which subset each thread must 
///                                      process at each round. </param>
/// <param name="numberOfRounds">        Number of rounds required to process the schedule. 
///                                      </param>
/// <param name="boundingBoxesMin">      Temporary array for the bounding boxes of each subset. 
///                                      </param>
/// <param name="boundingBoxesMax">      Temporary array for the bounding boxes of each subset. 
///                                      </param>
/// <param name="subsetAreas">           Temporary array for the areas of each subset. </param>
/// <param name="stackNode">             Node index stack used to modify the treelets. </param>
/// <param name="stackMask">             Subset mask stack used to modify the treelets. </param>
/// <param name="stackSize">             Stack size. Used for stackNode and StackMask. </param>
/// <param name="currentInternalNode">   Index to the treelet internal node that should be 
///                                      modified next. </param>
/// <param name="ci">                    Cost of traversing an internal node. </param>
/// <param name="ct">                    Cost of performing a ray-triangle intersection. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceTreeletReestructureOptimizer(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, int* subtreeTrianglesCount, float* nodesSah, int treeletSize, 
        int gamma, int* schedule, int numberOfRounds, float4* boundingBoxesMin, 
        float4* boundingBoxesMax, float* subsetAreas, int* stackNode, char* stackMask, 
        int* stackSize, int* currentInternalNode, float ci = 1.2f, float ct = 1.0f);

}