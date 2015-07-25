#pragma once

#include <vector>
#include <set>

namespace BVHRT
{

/// <summary> Generates schedules that will be used to process treelets. For more information, 
///           check <see cref="TRBVHOptimizer"/> </summary>.
///                                                      
/// <remarks> Leonardo, 1/2/2015. </remarks>
class TRBVHScheduler
{
public:

    /// <summary> Default constructor. </summary>
    ///
    /// <remarks> Leonardo, 1/2/2015. </remarks>
    TRBVHScheduler();

    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 1/2/2015. </remarks>
    ~TRBVHScheduler();

    /// <summary> Generates a schedule. 
    ///           
    ///           <para> The implemented algorithm was described in "KARRAS, T., AND AILA, T. 2013.
    ///           Fast parallel construction of high-quality bounding volume hierarchies.
    ///           In Proc. High-Performance Graphics." </para> </summary>
    ///
    /// <remarks> Leonardo, 1/2/2015. </remarks>
    ///
    /// <param name="treeletSize"> Treelet size. </param>
    /// <param name="warpSize">    Warp size. </param>
    /// <param name="schedule">    [out] The generated schedule. The outer vector represents round 
    ///                            and the inner one represents subsets. </param>
    void GenerateSchedule(int treeletSize, int warpSize, 
            std::vector<std::vector<int>>& schedule) const;

private:

    /// <summary> Process a subset to find its dependencies. </summary>
    ///
    /// <remarks> Leonardo, 1/2/2015. </remarks>
    ///
    /// <param name="treeletSize">    Treelet size. </param>
    /// <param name="warpSize">       Warp size. </param>
    /// <param name="subset">         The subset. </param>
    /// <param name="superset">       The set that was partitioned to find subset. </param>
    /// <param name="dependencies">   [in,out] The dependencies. </param>
    /// <param name="subsetsPerSize"> [in,out] Subsets grouped by size. </param>
    void ProcessSubset(int treeletSize, int warpSize, int subset, int superset,
            std::vector<std::set<int>>& dependencies, 
            std::vector<std::set<int>>& subsetsBySize) const;
};

}

