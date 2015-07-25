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

#include "ray/RayBufferKernels.hpp"

using namespace FW;

//------------------------------------------------------------------------

__device__ __inline__ void atomicMin(F32* ptr, F32 value)
{
    U32 curr = atomicAdd((U32*)ptr, 0);
    while (value < __int_as_float(curr))
    {
        U32 prev = curr;
        curr = atomicCAS((U32*)ptr, curr, __float_as_int(value));
        if (curr == prev)
            break;
    }
}

//------------------------------------------------------------------------

__device__ __inline__ void atomicMax(F32* ptr, F32 value)
{
    U32 curr = atomicAdd((U32*)ptr, 0);
    while (value > __int_as_float(curr))
    {
        U32 prev = curr;
        curr = atomicCAS((U32*)ptr, curr, __float_as_int(value));
        if (curr == prev)
            break;
    }
}

//------------------------------------------------------------------------

__device__ __inline__ void collectBits(volatile U32* hash, int index, U32 value)
{
    for (int i = 0; i < 32; i++)
        hash[(index + i * 6) >> 5] |= ((value >> i) & 1) << ((index + i * 6) & 31);
}

//------------------------------------------------------------------------

extern "C" __global__ void findAABBKernel(void)
{
    // Pick a bunch of rays for the block.

    const FindAABBInput& in = c_FindAABBInput;
    FindAABBOutput& out = *(FindAABBOutput*)&c_FindAABBOutput;

    int bidx        = blockIdx.x + blockIdx.y * gridDim.x;
    int tidx        = threadIdx.x + threadIdx.y * FindAABB_BlockWidth;
    int blockSize   = FindAABB_BlockWidth * FindAABB_BlockHeight;
    int blockStart  = bidx * blockSize * in.raysPerThread;
    int blockEnd    = ::min(blockStart + blockSize * in.raysPerThread, in.numRays);

    if (blockStart >= blockEnd)
        return;

    // Grow AABB by each thread.

    Vec3f threadLo(+FW_F32_MAX);
    Vec3f threadHi(-FW_F32_MAX);
    for (int i = blockStart + tidx; i < blockEnd; i += blockSize)
    {
        Ray ray = ((const Ray*)in.inRays)[i];
        threadLo = min(threadLo, ray.origin);
        threadHi = max(threadHi, ray.origin);
        ray.origin += ray.direction * ray.tmax;
        threadLo= min(threadLo, ray.origin);
        threadHi = max(threadHi, ray.origin);
    }

    // Perform reduction within the warp.

    __shared__ Vec3f redLo[FindAABB_BlockWidth * FindAABB_BlockHeight];
    __shared__ Vec3f redHi[FindAABB_BlockWidth * FindAABB_BlockHeight];
    redLo[tidx] = threadLo, redHi[tidx] = threadHi;
    redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 1]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 1]);
    redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 2]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 2]);
    redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 4]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 4]);
    redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 8]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 8]);
    redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 16]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 16]);

    // Perform reduction within the block.

    __syncthreads();
    if ((tidx & 32) == 0)
        redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 32]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 32]);

    __syncthreads();
    if ((tidx & 64) == 0)
        redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 64]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 64]);

    __syncthreads();
    if ((tidx & 128) == 0)
        redLo[tidx] = min(redLo[tidx], redLo[tidx ^ 128]), redHi[tidx] = max(redHi[tidx], redHi[tidx ^ 128]);

    // Accumulate globally.

    if (tidx == 0)
    {
        atomicMin(&out.aabbLo.x, redLo[tidx].x);
        atomicMin(&out.aabbLo.y, redLo[tidx].y);
        atomicMin(&out.aabbLo.z, redLo[tidx].z);
        atomicMax(&out.aabbHi.x, redHi[tidx].x);
        atomicMax(&out.aabbHi.y, redHi[tidx].y);
        atomicMax(&out.aabbHi.z, redHi[tidx].z);
    }
}

//------------------------------------------------------------------------

extern "C" __global__ void genMortonKeysKernel(void)
{
    // Get parameters.

    const GenMortonKeysInput& in = *(const GenMortonKeysInput*)c_GenMortonKeysInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= in.numRays)
        return;

    // Normalize origin and direction.

    const Ray& ray = ((const Ray*)in.inRays)[taskIdx];
    Vec3f a = (ray.origin - in.aabbLo) / (in.aabbHi - in.aabbLo);
    Vec3f b = (normalize(ray.direction) + 1.0f) * 0.5f;

    // Generate hash.

    __shared__ U32 hashbuf[32 * 4 * 6];
    volatile U32* hash = &hashbuf[(threadIdx.x + threadIdx.y * blockDim.x) * 6];
    for (int i = 0; i < 6; i++)
        hash[i] = 0;

    collectBits(hash, 0, (U32)(a.x * 256.0f * 65536.0f));
    collectBits(hash, 1, (U32)(a.y * 256.0f * 65536.0f));
    collectBits(hash, 2, (U32)(a.z * 256.0f * 65536.0f));
    collectBits(hash, 3, (U32)(b.x * 32.0f * 65536.0f));
    collectBits(hash, 4, (U32)(b.y * 32.0f * 65536.0f));
    collectBits(hash, 5, (U32)(b.z * 32.0f * 65536.0f));

    // Output key.

    MortonKey& key = ((MortonKey*)in.outKeys)[taskIdx];
    key.oldSlot = taskIdx;
    for (int i = 0; i < 6; i++)
        key.hash[i] = hash[i];
}

//------------------------------------------------------------------------

extern "C" __global__ void reorderRaysKernel(void)
{
    // Get parameters.

    const ReorderRaysInput& in = c_ReorderRaysInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= in.numRays)
        return;

    // Copy data.

    int oldSlot = ((const MortonKey*)in.inKeys)[taskIdx].oldSlot;
    int id = ((const S32*)in.inSlotToID)[oldSlot];

    ((Ray*)in.outRays)[taskIdx] = ((const Ray*)in.inRays)[oldSlot];
    ((S32*)in.outIDToSlot)[id] = taskIdx;
    ((S32*)in.outSlotToID)[taskIdx] = id;
}

//------------------------------------------------------------------------
