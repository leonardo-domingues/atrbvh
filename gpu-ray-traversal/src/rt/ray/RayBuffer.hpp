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
#include "gpu/Buffer.hpp"
#include "Util.hpp"

namespace FW
{

class RayBuffer
{
public:
                        RayBuffer               (S32 n=0, bool closestHit=true)     : m_size(0), m_needClosestHit(closestHit) { resize(n); }

    S32                 getSize                 () const                            { return m_size; }
    void                resize                  (S32 n);

    void                setRay                  (S32 slot, const Ray& ray)          { setRay(slot, ray, slot); }
    void                setRay                  (S32 slot, const Ray& ray, S32 id);
    void                setResult               (S32 slot, const RayResult& r)      { getMutableResultForSlot(slot) = r; }

//  const Ray&          operator[]              (S32 slot) const                    { return getRayForSlot(slot); }
    const Ray&          getRayForSlot           (S32 slot) const                    { FW_ASSERT(slot >= 0 && slot < m_size); return ((const Ray*)m_rays.getPtr())[slot]; }
    const Ray&          getRayForID             (S32 id) const                      { return getRayForSlot(getSlotForID(id)); }

    const RayResult&    getResultForSlot        (S32 slot) const                    { FW_ASSERT(slot >= 0 && slot < m_size); return ((const RayResult*)m_results.getPtr())[slot]; }
          RayResult&    getMutableResultForSlot (S32 slot)                          { FW_ASSERT(slot >= 0 && slot < m_size); return ((RayResult*)m_results.getMutablePtr())[slot]; }
    const RayResult&    getResultForID          (S32 id) const                      { return getResultForSlot(getSlotForID(id)); }
          RayResult&    getMutableResultForID   (S32 id)                            { return getMutableResultForSlot(getSlotForID(id)); }

    S32                 getSlotForID            (S32 id) const                      { FW_ASSERT(id >= 0 && id < m_size); return ((const S32*)m_IDToSlot.getPtr())[id]; }
    S32                 getIDForSlot            (S32 slot) const                    { FW_ASSERT(slot >= 0 && slot < m_size); return ((const S32*)m_slotToID.getPtr())[slot]; }

    void                setNeedClosestHit       (bool c)                            { m_needClosestHit = c; }
    bool                getNeedClosestHit       () const                            { return m_needClosestHit; }

    void                mortonSort              ();
    void                randomSort              (U32 randomSeed=0);

    Buffer&             getRayBuffer            ()                                  { return m_rays; }
    Buffer&             getResultBuffer         ()                                  { return m_results; }
    Buffer&             getIDToSlotBuffer       ()                                  { return m_IDToSlot; }
    Buffer&             getSlotToIDBuffer       ()                                  { return m_slotToID; }

private:
    S32                 m_size;
    mutable Buffer      m_rays;         // Ray
    mutable Buffer      m_results;      // RayResult
    mutable Buffer      m_IDToSlot;     // S32
    mutable Buffer      m_slotToID;     // S32

    bool                m_needClosestHit;
};

} //