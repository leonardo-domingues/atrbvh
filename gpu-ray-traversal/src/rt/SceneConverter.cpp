#include "SceneConverter.h"

#include <limits>
#include <algorithm>
#include <cuda_runtime.h>

using BVHRT::SceneWrapper;
using BVHRT::Scene;

SceneConverter::SceneConverter(FW::Scene* scene) : scene(scene)
{
    numberOfTriangles = scene->getNumTriangles();
}

SceneConverter::~SceneConverter()
{
}

SceneWrapper* SceneConverter::ConvertToCudaSceneDeviceMemory()
{
    FW::Buffer indexBuffer = scene->getTriVtxIndexBuffer();
    FW::Buffer vertexBuffer = scene->getVtxPosBuffer();

    const FW::Vec3i* indicesSource = (FW::Vec3i*)indexBuffer.getPtr();
    const FW::Vec3f* verticesSource = (FW::Vec3f*)vertexBuffer.getPtr();

    float4* vertices = new float4[numberOfTriangles * 3];
    float3 boundingBoxMin, boundingBoxMax;
    boundingBoxMin.x = std::numeric_limits<float>::max();
    boundingBoxMin.y = std::numeric_limits<float>::max();
    boundingBoxMin.z = std::numeric_limits<float>::max();
    boundingBoxMax.x = std::numeric_limits<float>::min();
    boundingBoxMax.y = std::numeric_limits<float>::min();
    boundingBoxMax.z = std::numeric_limits<float>::min();

    for (unsigned int i = 0; i < numberOfTriangles; ++i)
    {
        FW::Vec3i indices = indicesSource[i];

        FW::Vec3f vertex1 = verticesSource[indices.x];
        FW::Vec3f vertex2 = verticesSource[indices.y];
        FW::Vec3f vertex3 = verticesSource[indices.z];

        float4 v1;
        v1.x = vertex1.x;
        v1.y = vertex1.y;
        v1.z = vertex1.z;
        FindMinAndMax(v1, &boundingBoxMin, &boundingBoxMax);

        float4 v2;
        v2.x = vertex2.x;
        v2.y = vertex2.y;
        v2.z = vertex2.z;
        FindMinAndMax(v2, &boundingBoxMin, &boundingBoxMax);

        float4 v3;
        v3.x = vertex3.x;
        v3.y = vertex3.y;
        v3.z = vertex3.z;
        FindMinAndMax(v3, &boundingBoxMin, &boundingBoxMax);

        vertices[i * 3] = v1;
        vertices[i * 3 + 1] = v2;
        vertices[i * 3 + 2] = v3;
    }

    // Copy scene to GPU
    SceneWrapper* deviceScene = new SceneWrapper(numberOfTriangles, vertices, boundingBoxMin,
            boundingBoxMax);

    return deviceScene;
}

void SceneConverter::FindMinAndMax(float4 source, float3* min, float3* max)
{
    min->x = std::min(min->x, source.x);
    min->y = std::min(min->y, source.y);
    min->z = std::min(min->z, source.z);
    max->x = std::max(max->x, source.x);
    max->y = std::max(max->y, source.y);
    max->z = std::max(max->z, source.z);
}
