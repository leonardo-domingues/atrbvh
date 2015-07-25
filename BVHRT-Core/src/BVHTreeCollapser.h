#pragma once

#include "BVHCollapsedTree.h"
#include "BVHTree.h"

namespace BVHRT
{

/// <summary>  Calculate the SAH cost for each tree node, and also calculate the SAH cost if that 
///            node was a leaf referencing all triangles that descend from it. Collapses the 
///            subtree if that leads to a lower SAH cost than the original. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
class BVHTreeCollapser
{
public:

    /// <summary> Default constructor. </summary>
    ///
    /// <remarks> Leonardo, 02/11/2015. </remarks>
    BVHTreeCollapser();
    
    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 02/11/2015. </remarks>
    ~BVHTreeCollapser();

    /// <summary> Copy the elements that will not be changed when collapsing the tree from 
    ///           source to destination. </summary>
    ///
    /// <remarks> Leonardo, 02/11/2015. </remarks>
    ///
    /// <param name="deviceTree"> Source tree. </param>
    /// <param name="sah">        [out] Total tree BVH value. </param>
    /// <param name="ci">         Cost of traversing an internal node. </param>
    /// <param name="ct">         Cost of performing a ray-triangle intersection. </param>
    ///
    /// <returns> Pointer to the collapsed tree, allocated in device memory. </returns>
    BVHCollapsedTree* Collapse(BVHTree* deviceTree, float* sah, float ci = 1.2f, float ct = 1.0f);
};

}