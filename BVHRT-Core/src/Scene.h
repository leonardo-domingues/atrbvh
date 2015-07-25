#pragma once

#include <vector_types.h>

namespace BVHRT
{

/// <summary> Scene representation. This structure will be used mostly to send scene data to GPU.
///           </summary>
///
/// <remarks> Leonardo, 12/16/2014. </remarks>
struct Scene
{
    unsigned int numberOfTriangles;
    float4* vertices;
    float3 boundingBoxMin;
    float3 boundingBoxMax;
};

}
