#pragma once

#include <vector_types.h>
#include "BVHTree.h"
#include "Scene.h"

namespace BVHRT
{

/// <summary> Calculate 30-bit Morton codes for the scene triangles. </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="sortIndices">       [out] Sort indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        unsigned int* mortonCodes, unsigned int* sortIndices);

/// <summary> Calculate 63-bit Morton codes for the scene triangles. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="sortIndices">       [out] Sort indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        unsigned long long int* mortonCodes, unsigned int* sortIndices);

/// <summary> Calculate 30-bit Morton codes for the scene triangles when triangle splitting is
///           performed. </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="boundingBoxesMin">  [out] The bounding boxes minimum array. </param>
/// <param name="boundingBoxesMax">  [out] The bounding boxes maximum array. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="routingIndices">    [out] Routing indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax, unsigned int* mortonCodes, 
        unsigned int* routingIndices);

/// <summary> Calculate 63-bit Morton codes for the scene triangles when triangle splitting is
///           performed. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="boundingBoxesMin">  [out] The bounding boxes minimum array. </param>
/// <param name="boundingBoxesMax">  [out] The bounding boxes maximum array. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="routingIndices">    [out] Routing indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax,
        unsigned long long int* mortonCodes, unsigned int* routingIndices);

/// <summary> Build the BVH structure using 30-bit Morton codes. </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="sortedKeys">        The sorted Morton codes. </param>
/// <param name="sortIndices">       The sort indices. </param>
/// <param name="tree">              [out] BVH tree. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceBuildTree(unsigned int numberOfTriangles, unsigned int* sortedKeys,
        unsigned int* sortIndices, BVHTree* tree);

/// <summary> Build the BVH structure using 63-bit Morton codes. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="sortedKeys">        The sorted Morton codes. </param>
/// <param name="sortIndices">       The sort indices. </param>
/// <param name="tree">              [out] BVH tree. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceBuildTree(unsigned int numberOfTriangles, unsigned long long int* sortedKeys,
        unsigned int* sortIndices, BVHTree* tree);

/// <summary> Build the BVH structure when triangle splitting is performed, using 30-bit Morton 
///           codes. </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="sortedKeys">        The sorted Morton codes. </param>
/// <param name="dataIndices">       The data indices. </param>
/// <param name="tree">              [out] BVH tree. </param>
/// <param name="routingIndices">    The routing indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceBuildTree(unsigned int numberOfTriangles, unsigned int* sortedKeys,
        unsigned int* dataIndices, BVHTree* tree, unsigned int* routingIndices);
					  
/// <summary> Build the BVH structure when triangle splitting is performed, using 63-bit Morton 
///           codes. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="sortedKeys">        The sorted Morton codes. </param>
/// <param name="dataIndices">       The data indices. </param>
/// <param name="tree">              [out] BVH tree. </param>
/// <param name="routingIndices">    The routing indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceBuildTree(unsigned int numberOfTriangles, unsigned long long int* sortedKeys,
	    unsigned int* dataIndices, BVHTree* tree, unsigned int* routingIndices);

/// <summary> Calculate the bounding boxes of the tree's nodes, as well as their surface areas.
///           </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="tree">              [in,out] BVH tree. </param>
/// <param name="counters">          [in,out] The atomic counters. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceCalculateNodeBoundingBoxes(unsigned int numberOfTriangles, const Scene* scene,
        BVHTree* tree, unsigned int* counters);

/// <summary> Calculate the bounding boxes of the tree's internal nodes, as well as their surface
///           areas. </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="tree">              [in,out] BVH tree. </param>
/// <param name="counters">          [in,out] The atomic counters. </param>
/// <param name="leavesBboxMin">     Leaves bounding box minimum array. </param>
/// <param name="leavesBboxMax">     Leaves bounding box maximum array. </param>
/// <param name="routingIndices">    Routing indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceCalculateInternalNodeBoundingBoxes(unsigned int numberOfTriangles, BVHTree* tree,
        unsigned int* counters, float4* leavesBboxMin, float4* leavesBboxMax, 
        unsigned int* routingIndices);
}
