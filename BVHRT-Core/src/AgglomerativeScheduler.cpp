#include "AgglomerativeScheduler.h"

#include "Commons.cuh"

namespace BVHRT
{

AgglomerativeScheduler::AgglomerativeScheduler()
{
}

AgglomerativeScheduler::~AgglomerativeScheduler()
{
}

void AgglomerativeScheduler::GenerateScheduleUpper(int treeletSize, int warpSize,
        std::vector<int>& schedule) const
{
    int numberOfElements = sumArithmeticSequence(treeletSize - 1, 1, treeletSize - 1);
    int elementsPerWarp = 2 * warpSize;
    int numberOfIterations = (numberOfElements + elementsPerWarp - 1) / elementsPerWarp;
    int scheduleSize = numberOfIterations * warpSize;

    schedule.clear();
    schedule.resize(scheduleSize, 0);

    int count = 0;
    float multiplier = 0.0f;
    for (int i = 0; i < treeletSize; ++i)
    {
        for (int j = i + 1; j < treeletSize; ++j)
        {
            int index = count + static_cast<int>(multiplier)* warpSize;
            schedule[index] = (schedule[index] << 16) | (i << 8) | j;

            ++count;
            if (count == warpSize)
            {
                multiplier += 0.5f;
                count = 0;
            }
        }
    }
}

void AgglomerativeScheduler::GenerateScheduleLower(int treeletSize, int warpSize,
        std::vector<int>& schedule, int& scheduleSize) const
{
    int numberOfElements = sumArithmeticSequence(treeletSize - 1, 1, treeletSize - 1);
    int elementsPerWarp = 2 * warpSize;
    int numberOfIterations = (numberOfElements + elementsPerWarp - 1) / elementsPerWarp;
    scheduleSize = numberOfIterations * warpSize;

    schedule.clear();
    schedule.resize(scheduleSize, 0);

    int count = 0;
    float multiplier = 0.0f;
    for (int i = 0; i < treeletSize; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            int index = count + static_cast<int>(multiplier) * warpSize;
            schedule[index] = (schedule[index] << 16) | (i << 8) | j;

            ++count;
            if (count == warpSize)
            {
                multiplier += 0.5f;
                count = 0;
            }
        }
    }

    // If multiplier was not integer, shift the last 'warpSize' elements over to the left
    if (static_cast<int>(multiplier + 0.5f) > static_cast<int>(multiplier))
    {
        for (int i = count; i < 32; ++i)
        {
            int index = i + static_cast<int>(multiplier) * warpSize;
            schedule[index] = (schedule[index] << 16);
        }
    }

}

}

