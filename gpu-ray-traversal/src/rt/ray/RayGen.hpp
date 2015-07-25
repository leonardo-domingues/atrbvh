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

#pragma once
#include "base/Math.hpp"
#include "base/Array.hpp"
#include "gpu/CudaCompiler.hpp"
#include "ray/RayBuffer.hpp"
#include "ray/PixelTable.hpp"
#include "Scene.hpp"

namespace FW
{

class RayGen
{
public:
    RayGen(S32 maxBatchSize = 8*1024*1024);

    // true if batch continues
    void    primary(RayBuffer& orays, const Vec3f& origin, const Mat4f& nscreenToWorld, S32 w,S32 h,float maxDist);
    bool    shadow (RayBuffer& orays, RayBuffer& irays, int numSamples, const Vec3f& lightPos, float lightRadius, bool& newBatch, U32 randomSeed=0);
    bool    ao     (RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed=0); // non-const because of Buffer transfers

    // these are hack for various tests
    bool    random (RayBuffer& orays, const AABB& bounds, int numRays, bool closestHit, bool PosDir=false, U32 randomSeed=0);
    bool    random (RayBuffer& orays, const AABB& bounds, int numRays, bool closestHit, bool PosDir, bool& newBatch, U32 randomSeed=0);
    bool    randomReflection (RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed=0);

private:
    bool    batching(S32 numInputRays,S32 numSamples,S32& startIdx,bool& newBatch, S32& lo,S32& hi);

    S32             m_maxBatchSize;
    CudaCompiler    m_compiler;
    PixelTable      m_pixelTable;

    S32             m_shadowStartIdx;
    S32             m_aoStartIdx;
    S32             m_randomStartIdx;
};

} //
