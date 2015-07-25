#include "TRBVHScheduler.h"

#include <algorithm>

#define max(a, b) (a) > (b) ? (a) : (b)

inline int populationCount(int x)
{
    int count = 0;
    while (x > 0)
    {
        ++count;
        x &= x - 1;
    }
    return count;
}

namespace BVHRT
{

TRBVHScheduler::TRBVHScheduler()
{
}

TRBVHScheduler::~TRBVHScheduler()
{
}

void TRBVHScheduler::GenerateSchedule(int treeletSize, int warpSize,
    std::vector<std::vector<int>>& schedule) const
{
    int numberOfSubsets = (1 << treeletSize) - 1;

    schedule.clear();

    // Group of subsets that can be composed with each subset
    std::vector<std::set<int>> dependencies(numberOfSubsets + 1);

    // Subsets grouped by size
    std::vector<std::set<int>> subsetsPerSize(treeletSize - 1);

    // Round that each subset should be processed in
    std::vector<int> subsetRounds(numberOfSubsets + 1, -1);

    // Recursively process subsets, creating a list of dependencies that will be used to assemble 
    // the schedule
    ProcessSubset(treeletSize, warpSize, numberOfSubsets, 0, dependencies, subsetsPerSize);

    // Subset size
    for (int i = treeletSize - 2; i >= 0; --i)
    {
        // Subset
        for (auto& subset : subsetsPerSize[i])
        {
            unsigned int minimumRound = 0;

            // Check dependencies
            for (auto& dependency : dependencies[subset])
            {
                unsigned int dependencyRound = subsetRounds[dependency];
                minimumRound = max(minimumRound, dependencyRound + 1);
            }

            // Insert at a round
            unsigned int round = minimumRound;
            if (schedule.size() <= round)
            {
                schedule.push_back(std::vector<int>());
            }
            while (schedule[round].size() == warpSize)
            {
                ++round;
                if (schedule.size() <= round)
                {
                    schedule.push_back(std::vector<int>());
                }
            }
            schedule[round].push_back(subset);
            subsetRounds[subset] = round;
        }
    }

    for (size_t i = 0; i < schedule.size(); ++i)
    {
        for (size_t j = schedule[i].size(); j < static_cast<size_t>(warpSize); ++j)
        {
            schedule[i].push_back(0);
        }
    }

    std::reverse(schedule.begin(), schedule.end());
}

void TRBVHScheduler::ProcessSubset(int treeletSize, int warpSize, int subset, int superset,
    std::vector<std::set<int>>& dependencies, std::vector<std::set<int>>& subsetsBySize) const
{
    int subsetSize = populationCount(subset);
    if (subsetSize == 1)
    {
        return;
    }

    if (subsetSize <= treeletSize - 2)
    {
        dependencies[subset].insert(superset);
        subsetsBySize[subsetSize].insert(subset);
    }

    // Handle dependencies
    if (subsetSize > 2)
    {
        // Find each partition of the subset
        int delta = (subset - 1) & subset;
        int partition = (-delta) & subset;
        while (partition != 0)
        {
            int partitionComplement = partition ^ subset;

            ProcessSubset(treeletSize, warpSize, partition, subset, dependencies, subsetsBySize);
            ProcessSubset(treeletSize, warpSize, partitionComplement, subset, dependencies,
                subsetsBySize);

            partition = (partition - delta) & subset;
        }
    }
}

}
