#pragma once

#include "Scene.h"

namespace BVHRT
{

/// <summary> Wrapper for Scene objects. This wrapper holds two instances of the referenced Scene,
///           one in host memory and another in device memory. Both instances are released on the
///           destructor </summary>
///
/// <remarks> Leonardo, 12/16/2014. </remarks>
class SceneWrapper
{
public:

    /// <summary> Initializes both the host and device memory Scene instances. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <param name="numberOfTriangles"> Number of triangles. </param>
    /// <param name="vertices">          [in,out] The vertices. </param>
    /// <param name="boundingBoxMin">    The bounding box minimum. </param>
    /// <param name="boundingBoxMax">    The bounding box maximum. </param>
    SceneWrapper(unsigned int numberOfTriangles, float4* vertices, float3 boundingBoxMin,
                 float3 boundingBoxMax);

    /// <summary> Frees the host and device memory Scene instances. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ~SceneWrapper();

    /// <summary> Get a pointer to the host memory Scene. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <returns> A host memory Scene*. </returns>
    const Scene* HostScene() const;

    /// <summary> Get a pointer to the device memory Scene. </summary>
    ///
    /// <remarks> Leonardo, 12/16/2014. </remarks>
    ///
    /// <returns> A device memory Scene*. </returns>
    const Scene* DeviceScene() const;

private:
    Scene* hostScene;
    Scene* deviceScene;

    float4* deviceVertices;
};
}