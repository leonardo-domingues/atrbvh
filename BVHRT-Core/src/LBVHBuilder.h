#pragma once

#include "BVHBuilder.h"

namespace BVHRT
{

/// <summary> This builder creates BVH trees in GPU using the LBVH construction method described 
///           in "KARRAS, T. 2012. Maximizing parallelism in the construction of BVHs, octrees,
///           and k - d trees. In Proc. High-Performance Graphics, 33–37.". The tree can be 
///           generated using 32 or 64 bit Morton codes.
///
///           <para> It is also capable of splitting triangles using an algorithm implemented 
///                  based on the description found in "KARRAS, T., AND AILA, T. 2013. Fast 
///                  parallel construction of high-quality bounding volume hierarchies. In Proc. 
///                  High-Performance Graphics." </para>
/// </summary>
///
/// <remarks> Leonardo, 12/17/2014. </remarks>
class LBVHBuilder : public BVHBuilder
{
public:

    /// <summary> Default constructor. </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="use64BitMortonCodes"> true to indicate that 64 bit Morton codes should be 
    ///                                    used, false to indicate that 32 bit Morton codes 
	///                                    should be used.
    LBVHBuilder(bool use64BitMortonCodes = false);

    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ~LBVHBuilder();

    /// <summary> Builds a BVH tree using the method described in "KARRAS, T. 2012. Maximizing
    ///           parallelism in the construction of BVHs, octrees, and k - d trees. In Proc.
    ///           High-Performance Graphics, 33–37.". </summary>
    ///
    /// <remarks> Leonardo, 12/17/2014. </remarks>
    ///
    /// <param name="sceneWrapper"> The scene wrapper. </param>
    ///
    /// <returns> The constructed BVH tree, in device memory space. </returns>
    virtual BVHTree* BuildTree(const SceneWrapper* sceneWrapper) override;

private:

    bool use64BitMortonCodes;

    /// <summary> Builds tree with no triangle splitting. </summary>
    ///
    /// <remarks> Leonardo, 12/23/2014. </remarks>
    ///
    /// <param name="sceneWrapper"> The scene wrapper. </param>
    ///
    /// <returns> The built BVHTree. </returns>    
    template <typename T> BVHTree* BuildTreeNoTriangleSplitting(const SceneWrapper* sceneWrapper) 
            const;

};
}

