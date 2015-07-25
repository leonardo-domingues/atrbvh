#pragma once

#include "Scene.hpp"
#include "SceneWrapper.h"

class SceneConverter
{
public:
    SceneConverter(FW::Scene* scene);
    ~SceneConverter();

    BVHRT::SceneWrapper* ConvertToCudaSceneDeviceMemory();
    unsigned int GetNumberOfTriangles()
    {
        return numberOfTriangles;
    }

private:

    FW::Scene* scene;
    unsigned int numberOfTriangles;

    void FindMinAndMax(float4 source, float3* min, float3* max);
};

