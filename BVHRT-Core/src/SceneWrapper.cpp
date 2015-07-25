#include "SceneWrapper.h"

#include "CudaErrorCheck.h"

namespace BVHRT
{

SceneWrapper::SceneWrapper(unsigned int numberOfTriangles, float4* vertices, float3 boundingBoxMin,
                           float3 boundingBoxMax)
{
    // Allocate host scene

    hostScene = new Scene();
    hostScene->numberOfTriangles = numberOfTriangles;
    hostScene->vertices = vertices;
    hostScene->boundingBoxMin = boundingBoxMin;
    hostScene->boundingBoxMax = boundingBoxMax;

    // Allocate device scene
    Scene tempScene;
    tempScene.numberOfTriangles = numberOfTriangles;
    tempScene.boundingBoxMin = boundingBoxMin;
    tempScene.boundingBoxMax = boundingBoxMax;
    checkCudaError(cudaMalloc(&deviceVertices, numberOfTriangles * 3 * sizeof(float4)));
    checkCudaError(cudaMemcpy(deviceVertices, vertices, numberOfTriangles * 3 * sizeof(float4),
                              cudaMemcpyHostToDevice));
    tempScene.vertices = deviceVertices;
    checkCudaError(cudaMalloc(&deviceScene, sizeof(Scene)));
    checkCudaError(cudaMemcpy(deviceScene, &tempScene, sizeof(Scene), cudaMemcpyHostToDevice));
}

SceneWrapper::~SceneWrapper()
{
    // Free host scene
    delete[] hostScene->vertices;
    delete hostScene;

    // Free device scene
    checkCudaError(cudaFree(deviceVertices));
    checkCudaError(cudaFree(deviceScene));
}

const Scene* SceneWrapper::HostScene() const
{
    return hostScene;
}

const Scene* SceneWrapper::DeviceScene() const
{
    return deviceScene;
}
}

