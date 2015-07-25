/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cuda/Renderer.hpp"
#include "cuda/RendererKernels.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"

#include "AgglomerativeTreeletOptimizer.h"
#include "BVHRTSettings.h"
#include "BVHTree.h"
#include "BVHTreeCollapser.h"
#include "BVHTreeInstanceManager.h"
#include "LBVHBuilder.h"
#include "SceneConverter.h"
#include "TRBVHOptimizer.h"

#include <iostream>

using namespace FW;

//------------------------------------------------------------------------

Renderer::Renderer(void)
    :   m_raygen            (1 << 20),

        m_window            (NULL),
        m_enableRandom      (false),

        m_mesh              (NULL),
        m_scene             (NULL),
        m_bvh               (NULL),

        m_image             (NULL),
        m_cameraFar         (0.0f),

        m_newBatch          (true),
        m_batchRays         (NULL),
        m_batchStart        (0)
{
    m_compiler.setSourceFile("src/rt/cuda/RendererKernels.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/rt");
    m_compiler.include("src/framework");
    m_bvhCachePath = "bvhcache";

    m_platform = Platform("GPU");
    m_platform.setLeafPreferences(1, 8);
}

//------------------------------------------------------------------------

Renderer::~Renderer(void)
{
    setMesh(NULL);
    delete m_image;
}

//------------------------------------------------------------------------

void Renderer::setMesh(MeshBase* mesh)
{
    // Same mesh => done.

    if (mesh == m_mesh)
        return;

    // Deinit scene and BVH.

    delete m_scene;
    m_scene = NULL;
    invalidateBVH();

    // Create scene.

    m_mesh = mesh;
    if (mesh)
        m_scene = new Scene(*mesh);
}

//------------------------------------------------------------------------

void Renderer::setParams(const Params& params)
{
    m_params = params;
    m_tracer.setKernel(params.kernelName);
}

//------------------------------------------------------------------------

CudaBVH* Renderer::getCudaBVH(void)
{
    bool cacheEnabled = false;
    bool bvhrt = true;

    // BVH is already valid => done.

    BVHLayout layout = m_tracer.getDesiredBVHLayout();
    if (!m_mesh || (m_bvh && m_bvh->getLayout() == layout))
        return m_bvh;

    // Deinit.

    delete m_bvh;
    m_bvh = NULL;

    // Setup build parameters.

    BVH::Stats stats;
    m_buildParams.stats = &stats;

    // Determine cache file name.
    String cacheFileName = sprintf("%s/%08x.dat", m_bvhCachePath.getPtr(), hashBits(
                                       m_scene->hash(),
                                       m_platform.computeHash(),
                                       m_buildParams.computeHash(),
                                       layout));
    if (cacheEnabled)
    {
        // Cache file exists => import.

        if (!hasError())
        {
            File file(cacheFileName, File::Read);
            if (!hasError())
            {
                m_bvh = new CudaBVH(file);
                return m_bvh;
            }
            clearError();
        }
    }

    // Display status.
    printf("\nBuilding BVH...\nThis will take a while.\n");
    if (m_window)
        m_window->showModalMessage("Building BVH...");


    if (bvhrt)
    {
        BVHRTSettings settings;        
        SceneConverter converter(m_scene);
        BVHRT::SceneWrapper* sceneWrapper = converter.ConvertToCudaSceneDeviceMemory();

        // Build tree
        BVHRT::BVHTree* deviceTree = nullptr;
        if (settings.lbvh32)
        {
            BVHRT::LBVHBuilder builder;
            deviceTree = builder.BuildTree(sceneWrapper);
        }
        else if (settings.lbvh64)
        {
            BVHRT::LBVHBuilder builder(true);
            deviceTree = builder.BuildTree(sceneWrapper);
        }        

        // Optimize tree
        if (settings.trbvh)
        {
            BVHRT::TRBVHOptimizer trhvbOptimizer(settings.treeletSize, settings.iterations);
            trhvbOptimizer.Optimize(deviceTree);
        }
        if (settings.atrbvh)
        {
            BVHRT::AgglomerativeTreeletOptimizer agglomerativeOptimizer(settings.treeletSize, 
                    settings.iterations);
            agglomerativeOptimizer.Optimize(deviceTree);
        }

        // Convert tree to the format that will be used by the ray tracer
        BVHRT::BVHTreeInstanceManager instanceManager;
        if (settings.collapseTree)
        {
            float sah;
            BVHRT::BVHTreeCollapser collapser;
            BVHRT::BVHCollapsedTree* deviceCollapsed = collapser.Collapse(deviceTree, &sah);
            BVHRT::BVHCollapsedTree* hostCollapsed = 
                    instanceManager.DeviceToHostCollapsedTree(deviceCollapsed);
            m_bvh = new CudaBVH(hostCollapsed, m_scene, layout);
            delete hostCollapsed;
            instanceManager.FreeDeviceCollapsedTree(deviceCollapsed);
        }
        else
        {
            // Convert tree to the structure used by the ray tracer
            BVHRT::BVHTree* hostTree = instanceManager.DeviceToHostTree(deviceTree);
            std::cout << hostTree->SAH() << std::endl;
            m_bvh = new CudaBVH(hostTree->NumberOfTriangles(), hostTree, m_scene, layout);
            delete hostTree;
        }
        
        // Free memory        
        instanceManager.FreeDeviceTree(deviceTree);
        delete sceneWrapper;

        failIfError();
    }
    else
    {
        BVH bvh(m_scene, m_platform, m_buildParams);
        stats.print();
        m_bvh = new CudaBVH(bvh, layout);
        failIfError();
    }

    // Write to cache.
    if (cacheEnabled)
    {
        if (!hasError())
        {
            CreateDirectory(m_bvhCachePath.getPtr(), NULL);
            File file(cacheFileName, File::Create);
            m_bvh->serialize(file);
            clearError();
        }
    }

    // Display status.

    printf("Done.\n\n");
    return m_bvh;
}

//------------------------------------------------------------------------

F32 Renderer::renderFrame(GLContext* gl, const CameraControls& camera)
{
    F32 launchTime = 0.0f;
    beginFrame(gl, camera);
    while (nextBatch())
    {
        launchTime += traceBatch();
        updateResult();
    }
    displayResult(gl);
    return launchTime;
}

//------------------------------------------------------------------------

void Renderer::beginFrame(GLContext* gl, const CameraControls& camera)
{
    FW_ASSERT(gl && m_mesh);

    // Setup BVH.

    m_tracer.setBVH(getCudaBVH());

    // Setup result image.

    const Vec2i& size = gl->getViewSize();
    if (!m_image || m_image->getSize() != size)
    {
        delete m_image;
        m_image = new Image(size, ImageFormat::ABGR_8888);
        m_image->getBuffer().setHints(Buffer::Hint_CudaGL);
        m_image->clear();
    }

    // Generate primary rays.

    m_raygen.primary(m_primaryRays,
                     camera.getPosition(),
                     invert(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip()),
                     size.x, size.y,
                     camera.getFar());

    // Secondary rays enabled => trace primary rays.

    if (m_params.rayType != RayType_Primary)
        m_tracer.traceBatch(m_primaryRays);

    // Initialize state.

    m_cameraFar     = camera.getFar();
    m_newBatch      = true;
    m_batchRays     = NULL;
    m_batchStart    = 0;
}

//------------------------------------------------------------------------

bool Renderer::nextBatch(void)
{
    FW_ASSERT(m_scene);

    // Clean up the previous batch.

    if (m_batchRays)
        m_batchStart += m_batchRays->getSize();
    m_batchRays = NULL;

    // Generate new batch.

    U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
    switch (m_params.rayType)
    {
    case RayType_Primary:
        if (!m_newBatch)
            return false;
        m_newBatch = false;
        m_batchRays = &m_primaryRays;
        break;

    case RayType_AO:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_params.aoRadius,
                         m_newBatch, randomSeed))
            return false;
        m_batchRays = &m_secondaryRays;
        break;

    case RayType_Diffuse:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_cameraFar,
                         m_newBatch, randomSeed))
            return false;
        m_secondaryRays.setNeedClosestHit(true);
        m_batchRays = &m_secondaryRays;
        break;

    default:
        FW_ASSERT(false);
        return false;
    }

    // Sort rays.

    if (m_params.sortSecondary && m_params.rayType != RayType_Primary)
        m_batchRays->mortonSort();
    return true;
}

//------------------------------------------------------------------------

F32 Renderer::traceBatch(void)
{
    FW_ASSERT(m_batchRays);
    return m_tracer.traceBatch(*m_batchRays);
}

//------------------------------------------------------------------------

void Renderer::updateResult(void)
{
    FW_ASSERT(m_scene && m_image && m_batchRays);

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

    ReconstructInput& in    = *(ReconstructInput*)
                              module->getGlobal("c_ReconstructInput").getMutablePtr();
    in.numRaysPerPrimary    = (m_params.rayType == RayType_Primary) ? 1 : m_params.numSamples;
    in.firstPrimary         = m_batchStart / in.numRaysPerPrimary;
    in.numPrimary           = m_batchRays->getSize() / in.numRaysPerPrimary;
    in.isPrimary            = (m_params.rayType == RayType_Primary);
    in.isAO                 = (m_params.rayType == RayType_AO);
    in.isDiffuse            = (m_params.rayType == RayType_Diffuse);
    in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer().getCudaPtr();
    in.primaryResults       = m_primaryRays.getResultBuffer().getCudaPtr();
    in.batchIDToSlot        = m_batchRays->getIDToSlotBuffer().getCudaPtr();
    in.batchResults         = m_batchRays->getResultBuffer().getCudaPtr();
    in.triMaterialColor     = m_scene->getTriMaterialColorBuffer().getCudaPtr();
    in.triShadedColor       = m_scene->getTriShadedColorBuffer().getCudaPtr();
    in.pixels               = m_image->getBuffer().getMutableCudaPtr();

    // Launch.

    module->getKernel("reconstructKernel").launch(in.numPrimary);
}

//------------------------------------------------------------------------

void Renderer::displayResult(GLContext* gl)
{
    FW_ASSERT(gl);
    Mat4f oldXform = gl->setVGXform(Mat4f());
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    gl->drawImage(*m_image, Vec2f(0.0f), 0.5f, false);
    gl->setVGXform(oldXform);
    glPopAttrib();
}

//------------------------------------------------------------------------

int Renderer::getTotalNumRays(void)
{
    // Casting primary rays => no degenerates.

    if (m_params.rayType == RayType_Primary)
        return m_primaryRays.getSize();

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Set input and output.

    CountHitsInput& in = *(CountHitsInput*)module->getGlobal("c_CountHitsInput").getMutablePtr();
    in.numRays = m_primaryRays.getSize();
    in.rayResults = m_primaryRays.getResultBuffer().getCudaPtr();
    in.raysPerThread = 32;
    module->getGlobal("g_CountHitsOutput").clear();

    // Count primary ray hits.

    module->getKernel("countHitsKernel").launch(
        (in.numRays - 1) / in.raysPerThread + 1,
        Vec2i(CountHits_BlockWidth, CountHits_BlockHeight));

    int numHits = *(S32*)module->getGlobal("g_CountHitsOutput").getPtr();

    // numSecondary = secondaryPerPrimary * primaryHits

    return numHits * m_params.numSamples;
}

//------------------------------------------------------------------------
