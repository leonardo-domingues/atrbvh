#pragma once

#include "BVHOptimizer.h"

namespace BVHRT
{

/// <summary> Optimize a BVH tree using the TRBVH method, described in detail in "KARRAS, T., 
///           AND AILA, T. 2013. Fast parallel construction of high-quality bounding volume 
///           hierarchies. In Proc. High-Performance Graphics.". 
///      
///           <para>We optimized the TRBVH kernels to use 75% occupancy, instead of 100% reported 
///           in the original paper. Memory allocations and the resource that is used to store 
///           each value should also be slightly different.</para>
///           </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
class TRBVHOptimizer : public BVHOptimizer
{
public:
    /// <summary> Constructor. </summary>
    ///
    /// <remarks> Leonardo, 02/11/2015. </remarks>
    ///
    /// <param name="treeletSize"> Treelet size. </param>
    /// <param name="iterations">  Number of iterations. </param>
    TRBVHOptimizer(int treeletSize = 7, int iterations = 3);

    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 02/11/2015. </remarks>
    ~TRBVHOptimizer();

    /// <summary> Optimizes the tree using the TRBVH algorithm. </summary>
    ///
    /// <remarks> Leonardo, 02/11/2015. </remarks>
    ///
    /// <param name="deviceTree"> [in,out] The BVH tree. </param>
    virtual void Optimize(BVHTree* deviceTree) override;

private:
    int treeletSize;
    int iterations;
};

}
