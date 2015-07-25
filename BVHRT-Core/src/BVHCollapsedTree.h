#pragma once

#include <vector_types.h>

namespace BVHRT
{

// Structure of arrays
struct BVHCollapsedNodes
{
    int* parentIndices;
    int* leftIndices;
    int* rightIndices;
    int* dataIndices;
    int* triangleCount;
    float4* boundingBoxMin;
    float4* boundingBoxMax;
};

/// <summary> Collapse a BVH tree by putting more than one triangle at each leaf, in order to 
///           minimize the overall tree SAH cost. For each internal node, analyze if its cost as
///           a subtree root is less than its cost as a leaf node; if it is not, collapse that 
///           subtree.
///      
///           <para>Tree collapsing is performed as a post-processing, after the tree structure 
///           has already been optimized.</para>
///           </summary>
///
/// <remarks> Leonardo, 04/04/2015. </remarks>
class BVHCollapsedTree
{   
public:

    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ~BVHCollapsedTree();

    friend class BVHTreeInstanceManager;

    int* triangleIndices;
    BVHCollapsedNodes nodes;
    int rootIndex;

private:

    /// <summary> Constructor. </summary>
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ///
    /// <param name="numberOfTriangles"> Number of triangles in the tree. </param>
    /// <param name="allocateMemory">    If true, host memory is allocated for the structure. 
    ///                                  If false, no memory is allocated. </param>
    BVHCollapsedTree(unsigned int numberOfTriangles, bool allocateMemory = true);

    unsigned int numberOfTriangles;
    
};

}
