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

enum
{
    FindAABB_BlockWidth     = 32,
    FindAABB_BlockHeight    = 8,
};

//------------------------------------------------------------------------

struct MortonKey
{
    S32         oldSlot;
    U32         hash[6];        // 192-bit Morton key
};

//------------------------------------------------------------------------

struct FindAABBInput
{
    S32         numRays;
    S32         raysPerThread;
    CUdeviceptr inRays;         // const Ray*
};

//------------------------------------------------------------------------

struct FindAABBOutput
{
    Vec3f       aabbLo;
    Vec3f       aabbHi;
};

//------------------------------------------------------------------------

struct GenMortonKeysInput
{
    S32         numRays;
    Vec3f       aabbLo;
    Vec3f       aabbHi;
    CUdeviceptr inRays;         // const Ray*
    CUdeviceptr outKeys;        // MortonKey*
};

//------------------------------------------------------------------------

struct ReorderRaysInput
{
    S32         numRays;
    CUdeviceptr inKeys;         // const MortonKey*
    CUdeviceptr inRays;         // const Ray*
    CUdeviceptr inSlotToID;     // const S32*
    CUdeviceptr outRays;        // Ray*
    CUdeviceptr outIDToSlot;    // S32*
    CUdeviceptr outSlotToID;    // S32*
};

//------------------------------------------------------------------------

#if FW_CUDA
extern "C"
{

__constant__ FindAABBInput c_FindAABBInput;
__device__ int4 c_FindAABBOutput[(sizeof(FindAABBOutput) + sizeof(int4) - 1) / sizeof(int4)];
__global__ void findAABBKernel(void);

__constant__ int4 c_GenMortonKeysInput[(sizeof(GenMortonKeysInput) + sizeof(int4) - 1) / sizeof(int4)];
__global__ void genMortonKeysKernel(void);

__constant__ ReorderRaysInput c_ReorderRaysInput;
__global__ void reorderRaysKernel(void);

}
#endif

//------------------------------------------------------------------------
}
