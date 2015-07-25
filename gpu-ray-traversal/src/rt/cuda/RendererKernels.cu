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

#include "cuda/RendererKernels.hpp"
#include "base/Math.hpp"

using namespace FW;

//------------------------------------------------------------------------

__device__ inline Vec4f fromABGR(U32 abgr)
{
    return Vec4f(
        (F32)(abgr & 0xFF) * (1.0f / 255.0f),
        (F32)((abgr >> 8) & 0xFF) * (1.0f / 255.0f),
        (F32)((abgr >> 16) & 0xFF) * (1.0f / 255.0f),
        (F32)(abgr >> 24) * (1.0f / 255.0f));
}

//------------------------------------------------------------------------

__device__ inline U32 toABGR(Vec4f v)
{
    return
        (U32)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 255.0f) |
        ((U32)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 255.0f) << 8) |
        ((U32)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 255.0f) << 16) |
        ((U32)(fminf(fmaxf(v.w, 0.0f), 1.0f) * 255.0f) << 24);
}

//------------------------------------------------------------------------

extern "C" __global__ void reconstructKernel(void)
{
    // Get parameters.

    const ReconstructInput& in = c_ReconstructInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= in.numPrimary)
        return;

    // Initialize.

    int                     primarySlot     = in.firstPrimary + taskIdx;
    int                     primaryID       = ((const S32*)in.primarySlotToID)[primarySlot];
    const RayResult&        primaryResult   = ((const RayResult*)in.primaryResults)[primarySlot];
    const S32*              batchSlots      = (const S32*)in.batchIDToSlot + ((in.isPrimary) ? primaryID : taskIdx * in.numRaysPerPrimary);
    const RayResult*        batchResults    = (const RayResult*)in.batchResults;
    const U32*          triMaterialColor    = (const U32*)in.triMaterialColor;
    const U32*          triShadedColor      = (const U32*)in.triShadedColor;
    U32&                    pixel           = ((U32*)in.pixels)[primaryID];
    Vec4f                   bgColor         = Vec4f(0.2f, 0.4f, 0.8f, 1.0f);

    // Accumulate color from each ray in the batch.

    Vec4f color = Vec4f(0.0f);
    for (int i = 0; i < in.numRaysPerPrimary; i++)
    {
        int tri = batchResults[batchSlots[i]].id;					// hit index
        if (tri == -1)
		{
			if(in.isPrimary)	color += bgColor;					// Primary: missed the scene, use background color
			else				color += Vec4f(1.0f);				// AO: not blocked, use white (should be light color). Arbitrary choice for Diffuse.
		}
        else
		{
			if(in.isAO)			color += Vec4f(0,0,0,1);			// AO: blocked, use white
			else				color += fromABGR(triShadedColor[tri]);
		}
    }
    color *= 1.0f / (F32)in.numRaysPerPrimary;

    // Diffuse: modulate with primary hit color.

    int tri = primaryResult.id;
    if (in.isAO && tri == -1)   color = bgColor;
    if (in.isDiffuse)			color *= (tri == -1) ? bgColor : fromABGR(triMaterialColor[tri]);

    // Write result.

    pixel = toABGR(color);
}

//------------------------------------------------------------------------

extern "C" __global__ void countHitsKernel(void)
{
    // Pick a bunch of rays for the block.

    const CountHitsInput& in = c_CountHitsInput;

    int bidx        = blockIdx.x + blockIdx.y * gridDim.x;
    int tidx        = threadIdx.x + threadIdx.y * CountHits_BlockWidth;
    int blockSize   = CountHits_BlockWidth * CountHits_BlockHeight;
    int blockStart  = bidx * blockSize * in.raysPerThread;
    int blockEnd    = ::min(blockStart + blockSize * in.raysPerThread, in.numRays);

    if (blockStart >= blockEnd)
        return;

    // Count hits by each thread.

    S32 threadTotal = 0;
    for (int i = blockStart + tidx; i < blockEnd; i += blockSize)
        if (((const RayResult*)in.rayResults)[i].id >= 0)
            threadTotal++;

    // Perform reduction within the warp.

    __shared__ volatile S32 red[CountHits_BlockWidth * CountHits_BlockHeight];
    red[tidx] = threadTotal;
    red[tidx] += red[tidx ^ 1];
    red[tidx] += red[tidx ^ 2];
    red[tidx] += red[tidx ^ 4];
    red[tidx] += red[tidx ^ 8];
    red[tidx] += red[tidx ^ 16];

    // Perform reduction within the block.

    __syncthreads();
    if ((tidx & 32) == 0)
        red[tidx] += red[tidx ^ 32];

    __syncthreads();
    if ((tidx & 64) == 0)
        red[tidx] += red[tidx ^ 64];

    __syncthreads();
    if ((tidx & 128) == 0)
        red[tidx] += red[tidx ^ 128];

    // Accumulate globally.

    if (tidx == 0)
        atomicAdd(&g_CountHitsOutput, red[tidx]);
}

//------------------------------------------------------------------------
