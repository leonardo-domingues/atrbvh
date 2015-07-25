#pragma once

#include <vector>

namespace BVHRT
{

/// <summary> Generates a schedule containing all pairs of clusters in an order such that storing 
///           their distances can be done in coalesced memory transactions. </summary>
///                                                      
/// <remarks> Leonardo, 04/04/2015. </remarks>
class AgglomerativeScheduler
{
public:

    /// <summary> Default constructor. </summary>
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    AgglomerativeScheduler();

    /// <summary> Destructor. </summary>
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ~AgglomerativeScheduler();

    /// <summary> Generates a schedule for an upper triangular matrix.
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ///
    /// <param name="treeletSize"> Treelet size. </param>
    /// <param name="warpSize">    Warp size. </param>
    /// <param name="schedule">    [out] The generated schedule. The outer vector represents round 
    ///                            and the inner one represents subsets. </param>
    void GenerateScheduleUpper(int treeletSize, int warpSize,
        std::vector<int>& schedule) const;

    /// <summary> Generates a schedule for an upper triangular matrix.
    ///
    /// <remarks> Leonardo, 04/04/2015. </remarks>
    ///
    /// <param name="treeletSize">  Treelet size. </param>
    /// <param name="warpSize">     Warp size. </param>
    /// <param name="schedule">     [out] The generated schedule. The outer vector represents 
    ///                             round and the inner one represents subsets. </param>
    /// <param name="scheduleSize"> Number of elements contained in the schedule. </param>
    void GenerateScheduleLower(int treeletSize, int warpSize,
        std::vector<int>& schedule, int& scheduleSize) const;
};

}
