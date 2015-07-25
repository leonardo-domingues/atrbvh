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
#include "base/DLLImports.hpp"
#include "base/Math.hpp"
#include "Util.hpp"

namespace FW
{
//------------------------------------------------------------------------

struct RayGenPrimaryInput
{
    Vec3f       origin;
    Mat4f       nscreenToWorld;
    S32         w;
    S32         h;
    F32         maxDist;
    CUdeviceptr rays;           // Ray*
    CUdeviceptr idToSlot;       // S32*
    CUdeviceptr slotToID;       // S32*
    CUdeviceptr indexToPixel;   // const S32*
};

//------------------------------------------------------------------------

struct RayGenAOInput
{
    S32         firstInputSlot;
    S32         numInputRays;
    S32         numSamples;
    F32         maxDist;
    U32         randomSeed;
    CUdeviceptr inRays;         // const Ray*
    CUdeviceptr inResults;      // const RayResult*
    CUdeviceptr outRays;        // Ray*
    CUdeviceptr outIDToSlot;    // S32*
    CUdeviceptr outSlotToID;    // S32*
    CUdeviceptr normals;        // const Vec3f*
};

//------------------------------------------------------------------------

struct RayGenShadowInput
{
    S32         firstInputSlot;
    S32         numInputRays;
    S32         numSamples;
	F32			lightPositionX;
	F32			lightPositionY;
	F32			lightPositionZ;
    F32         lightRadius;
    U32         randomSeed;
    CUdeviceptr inRays;         // const Ray*
    CUdeviceptr inResults;      // const RayResult*
    CUdeviceptr outRays;        // Ray*
    CUdeviceptr outIDToSlot;    // S32*
    CUdeviceptr outSlotToID;    // S32*
};

//------------------------------------------------------------------------

#if FW_CUDA
extern "C"
{

__constant__ int4 c_RayGenPrimaryInput[(sizeof(RayGenPrimaryInput) + sizeof(int4) - 1) / sizeof(int4)];
__global__ void rayGenPrimaryKernel(void);

__constant__ RayGenAOInput c_RayGenAOInput;
__global__ void rayGenAOKernel(void);

__constant__ RayGenShadowInput c_RayGenShadowInput;
__global__ void rayGenShadowKernel(void);

}
#endif

//------------------------------------------------------------------------
}
