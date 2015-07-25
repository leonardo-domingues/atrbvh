#pragma once

#include "BVHTree.h"

namespace BVHRT
{

/// <summary> BVHOptimizer serves as a common interface for classes implementing different 
///           algorithms for optimizing BVH structures. </summary>
///
/// <remarks> Leonardo, 12/25/2014. </remarks>
class BVHOptimizer
{
public:
    /// <summary> Optimizes the given device tree. </summary>
    ///
    /// <remarks> Leonardo, 12/25/2014. </remarks>
    ///
    /// <param name="deviceTree"> [in,out] The device tree. </param>
    virtual void Optimize(BVHTree* deviceTree) = 0;
};
}
