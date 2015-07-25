#pragma once

#include "BVHOptimizer.h"

namespace BVHRT
{

/// <summary> Optimize a BVH tree using the ATRBVH method, described in detail in "DOMINGUES, L., 
///           AND PEDRINI, H. 2015. Bounding volume hierarchy optimization through agglomerative 
///           treelet restructuring. Accepted to High-Performance Graphics 2015.". 
///           </summary>
///
/// <remarks> Leonardo, 04/04/2015. </remarks>
class AgglomerativeTreeletOptimizer : public BVHOptimizer
{
public:

    /// <summary> Constructor. </summary>
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ///
    /// <param name="treeletSize"> Treelet size. </param>
    /// <param name="iterations">  Number of iterations. </param>
    AgglomerativeTreeletOptimizer(int treeletSize = 9, int iterations = 2);

    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ~AgglomerativeTreeletOptimizer();

    /// <summary> Optimizes the tree using the ATRBVH algorithm. </summary>
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ///
    /// <param name="deviceTree"> [in,out] The BVH tree. </param>
    virtual void Optimize(BVHTree* deviceTree) override;

private:
    int treeletSize;
    int iterations;
};
}
