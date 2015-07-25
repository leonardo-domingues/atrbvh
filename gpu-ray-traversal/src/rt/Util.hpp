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

namespace FW
{
//------------------------------------------------------------------------

class AABB
{
public:
    FW_CUDA_FUNC                    AABB        (void) : m_mn(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX), m_mx(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX) {}
    FW_CUDA_FUNC                    AABB        (const Vec3f& mn, const Vec3f& mx) : m_mn(mn), m_mx(mx) {}

    FW_CUDA_FUNC    void            grow        (const Vec3f& pt)   { m_mn = m_mn.min(pt); m_mx = m_mx.max(pt); }
    FW_CUDA_FUNC    void            grow        (const AABB& aabb)  { grow(aabb.m_mn); grow(aabb.m_mx); }
    FW_CUDA_FUNC    void            intersect   (const AABB& aabb)  { m_mn = m_mn.max(aabb.m_mn); m_mx = m_mx.min(aabb.m_mx); }
    FW_CUDA_FUNC    float           volume      (void) const        { if(!valid()) return 0.0f; return (m_mx.x-m_mn.x) * (m_mx.y-m_mn.y) * (m_mx.z-m_mn.z); }
    FW_CUDA_FUNC    float           area        (void) const        { if(!valid()) return 0.0f; Vec3f d = m_mx - m_mn; return (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f; }
    FW_CUDA_FUNC    bool            valid       (void) const        { return m_mn.x<=m_mx.x && m_mn.y<=m_mx.y && m_mn.z<=m_mx.z; }
    FW_CUDA_FUNC    Vec3f           midPoint    (void) const        { return (m_mn+m_mx)*0.5f; }
    FW_CUDA_FUNC    const Vec3f&    min         (void) const        { return m_mn; }
    FW_CUDA_FUNC    const Vec3f&    max         (void) const        { return m_mx; }
    FW_CUDA_FUNC    Vec3f&          min         (void)              { return m_mn; }
    FW_CUDA_FUNC    Vec3f&          max         (void)              { return m_mx; }

    FW_CUDA_FUNC    AABB            operator+   (const AABB& aabb) const { AABB u(*this); u.grow(aabb); return u; }

private:
    Vec3f           m_mn;
    Vec3f           m_mx;
};

//------------------------------------------------------------------------

struct Ray
{
    FW_CUDA_FUNC            Ray         (void)      : origin(0.0f), tmin(0.0f), direction(0.0f), tmax(0.0f) {}
    FW_CUDA_FUNC    void    degenerate  (void)      { tmax = tmin - 1.0f; }

    Vec3f           origin;
    float           tmin;
    Vec3f           direction;
    float           tmax;
};

//------------------------------------------------------------------------

#define RAY_NO_HIT  (-1)

struct RayResult
{
    FW_CUDA_FUNC            RayResult   (S32 ii = RAY_NO_HIT, float ti = 0.f) : id(ii), t(ti) {}
    FW_CUDA_FUNC    bool    hit         (void) const    { return (id != RAY_NO_HIT); }
    FW_CUDA_FUNC    void    clear       (void)          { id = RAY_NO_HIT; }

    S32             id;
    float           t;
    S32             padA;
    S32             padB;
};

//------------------------------------------------------------------------

namespace Intersect
{
    Vec2f RayBox(const AABB& box, const Ray& ray);
    Vec3f RayTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Ray& ray);
    Vec3f RayTriangleWoop(const Vec4f& zpleq, const Vec4f& upleq, const Vec4f& vpleq, const Ray& ray);
}

//------------------------------------------------------------------------
}