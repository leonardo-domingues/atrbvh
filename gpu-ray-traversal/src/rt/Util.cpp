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

#include "Util.hpp"

using namespace FW;

//------------------------------------------------------------------------

Vec2f Intersect::RayBox(const AABB& box, const Ray& ray)
{
    const Vec3f& orig = ray.origin;
    const Vec3f& dir  = ray.direction;

    Vec3f t0 = (box.min() - orig) / dir;
    Vec3f t1 = (box.max() - orig) / dir;

    float tmin = min(t0,t1).max();
    float tmax = max(t0,t1).min();

    return Vec2f(tmin,tmax);
}

//------------------------------------------------------------------------

Vec3f Intersect::RayTriangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Ray& ray)
{
//  const float EPSILON = 0.000001f; // breaks FairyForest
    const float EPSILON = 0.f; // works better
    const Vec3f miss(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);

    Vec3f edge1 = v1-v0;
    Vec3f edge2 = v2-v0;
    Vec3f pvec  = cross(ray.direction,edge2);
    float det   = dot(edge1, pvec);

    Vec3f tvec = ray.origin - v0;
    float u = dot(tvec,pvec);

    Vec3f qvec = cross(tvec, edge1);
    float v = dot(ray.direction, qvec);

    // TODO: clear this
    if (det > EPSILON)
    {
        if (u < 0.0 || u > det)
            return miss;
        if (v < 0.0 || u + v > det)
            return miss;
    }
    else if(det < -EPSILON)
    {
        if (u > 0.0 || u < det)
            return miss;
        if (v > 0.0 || u + v < det)
            return miss;
    }
    else
        return miss;

    float inv_det = 1.f / det;
    float t = dot(edge2, qvec) * inv_det;
    u *= inv_det;
    v *= inv_det;

    if(t>ray.tmin && t<ray.tmax)
        return Vec3f(u,v,t);

    return miss;
}


//------------------------------------------------------------------------

Vec3f Intersect::RayTriangleWoop(const Vec4f& zpleq, const Vec4f& upleq, const Vec4f& vpleq, const Ray& ray)
{
    const Vec3f miss(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);

    Vec4f orig(ray.origin,1.f);
    Vec4f dir (ray.direction,0.f);

    float Oz   = dot(zpleq,orig);           // NOTE: differs from HPG kernels!
    float ooDz = 1.f / dot(dir,zpleq);
    float t = -Oz * ooDz;
    if (t>ray.tmin && t<ray.tmax)
    {
        float Ou = dot(upleq,orig);
        float Du = dot(upleq,dir);
        float u = Ou + t*Du;
        if (u >= 0)
        {
            float Ov = dot(vpleq,orig);
            float Dv = dot(vpleq,dir);
            float v = Ov + t*Dv;
            if (v >= 0 && (u+v) <= 1.f)
            {
                return Vec3f(u,v,t);
            }
        }
    }

    return miss;
}

//------------------------------------------------------------------------
