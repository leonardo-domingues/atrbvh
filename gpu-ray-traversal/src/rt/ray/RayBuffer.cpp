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

#include "ray/RayBuffer.hpp"
#include "ray/RayBufferKernels.hpp"
#include "base/Math.hpp"
#include "base/Sort.hpp"
#include "base/Random.hpp"
#include "gpu/CudaCompiler.hpp"

namespace FW
{

void RayBuffer::resize(S32 n)
{
    FW_ASSERT(n >= 0);
    if (n < m_size)
    {
        m_size = n;
        return;
    }

    m_size = n;
    m_rays.resize(n * sizeof(Ray));
    m_results.resize(n * sizeof(RayResult));
    m_IDToSlot.resize(n * sizeof(S32));
    m_slotToID.resize(n * sizeof(S32));
}

void RayBuffer::setRay(S32 slot, const Ray& ray, S32 id)
{
    FW_ASSERT(slot >= 0 && slot < m_size);
    FW_ASSERT(id >= 0 && id < m_size);

    ((Ray*)m_rays.getMutablePtr())[slot] = ray;
    ((S32*)m_IDToSlot.getMutablePtr())[id] = slot;
    ((S32*)m_slotToID.getMutablePtr())[slot] = id;
}

//-------------------------------------------------------------------

void RayBuffer::randomSort(U32 randomSeed)
{
    // Reorder rays.

    Ray* rays = (Ray*)getRayBuffer().getMutablePtr();
    S32* idToSlot = (S32*)getIDToSlotBuffer().getMutablePtr();
    S32* slotToID = (S32*)getSlotToIDBuffer().getMutablePtr();
    Random random(randomSeed);

    for (int slot = 0; slot < m_size; slot++)
    {
        S32 slot2 = random.getS32(m_size - slot) + slot;

        S32 id  = slotToID[slot];
        S32 id2 = slotToID[slot2];

        swap(rays[slot],    rays[slot2]);
        swap(slotToID[slot],slotToID[slot2]);
        swap(idToSlot[id],  idToSlot[id2]);
    }
}

//-------------------------------------------------------------------

static bool compareMortonKey(void* data, int idxA, int idxB)
{
    const MortonKey& a = ((const MortonKey*)data)[idxA];
    const MortonKey& b = ((const MortonKey*)data)[idxB];
    if (a.hash[5] != b.hash[5]) return (a.hash[5] < b.hash[5]);
    if (a.hash[4] != b.hash[4]) return (a.hash[4] < b.hash[4]);
    if (a.hash[3] != b.hash[3]) return (a.hash[3] < b.hash[3]);
    if (a.hash[2] != b.hash[2]) return (a.hash[2] < b.hash[2]);
    if (a.hash[1] != b.hash[1]) return (a.hash[1] < b.hash[1]);
    if (a.hash[0] != b.hash[0]) return (a.hash[0] < b.hash[0]);
    return false;
}

void RayBuffer::mortonSort()
{
    // Compile kernels.

    CudaCompiler compiler;
    compiler.setSourceFile("src/rt/ray/RayBufferKernels.cu");
    compiler.addOptions("-use_fast_math");
    compiler.include("src/rt");
    compiler.include("src/framework");
    CudaModule* module = compiler.compile();

    // Allocate temporary buffers.

    Buffer oldRayBuffer = getRayBuffer();
    Buffer oldSlotToIDBuffer = getSlotToIDBuffer();
    Buffer keyBuffer(NULL, getSize() * sizeof(MortonKey));

    // Find AABB of the rays.
    {
        FindAABBInput& in = *(FindAABBInput*)module->getGlobal("c_FindAABBInput").getMutablePtr();
        FindAABBOutput& out = *(FindAABBOutput*)module->getGlobal("c_FindAABBOutput").getMutablePtr();
        in.numRays          = getSize();
        in.inRays           = getRayBuffer().getCudaPtr();
        in.raysPerThread    = 32;
        out.aabbLo          = Vec3f(+FW_F32_MAX);
        out.aabbHi          = Vec3f(-FW_F32_MAX);

        module->getKernel("findAABBKernel").launch(
            (in.numRays - 1) / in.raysPerThread + 1,
            Vec2i(FindAABB_BlockWidth, FindAABB_BlockHeight));
    }

    // Generate keys.
    {
        const FindAABBOutput& aabb = *(const FindAABBOutput*)module->getGlobal("c_FindAABBOutput").getPtr();
        GenMortonKeysInput& in = *(GenMortonKeysInput*)module->getGlobal("c_GenMortonKeysInput").getMutablePtr();
        in.numRays  = getSize();
        in.aabbLo   = aabb.aabbLo;
        in.aabbHi   = aabb.aabbHi;
        in.inRays   = getRayBuffer().getCudaPtr();
        in.outKeys  = keyBuffer.getMutableCudaPtr();
        module->getKernel("genMortonKeysKernel").launch(getSize(), Vec2i(32, 4));
    }

    // Sort keys.

    sort((MortonKey*)keyBuffer.getMutablePtr(), getSize(), compareMortonKey, sortDefaultSwap<MortonKey>, true);

    // Reorder rays.
    {
        ReorderRaysInput& in = *(ReorderRaysInput*)module->getGlobal("c_ReorderRaysInput").getMutablePtr();
        in.numRays      = getSize();
        in.inKeys       = keyBuffer.getCudaPtr();
        in.inRays       = oldRayBuffer.getCudaPtr();
        in.inSlotToID   = oldSlotToIDBuffer.getCudaPtr();
        in.outRays      = getRayBuffer().getMutableCudaPtr();
        in.outIDToSlot  = getIDToSlotBuffer().getMutableCudaPtr();
        in.outSlotToID  = getSlotToIDBuffer().getMutableCudaPtr();
        module->getKernel("reorderRaysKernel").launch(getSize());
    }
}


} //
