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

#include "ray/RayGenKernels.hpp"

using namespace FW;

//------------------------------------------------------------------------

__device__ inline void jenkinsMix(U32& a, U32& b, U32& c)
{
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);	// ~36 instructions
}

__device__ inline float hammersley(int i, int num)
{
	return (i+0.5f) / num;
}

__device__ inline Vec2f sobol2D(int i)
{
	Vec2f result;
	// remaining components by matrix multiplication 
	unsigned int r1 = 0, r2 = 0; 
	for (unsigned int v1 = 1U << 31, v2 = 3U << 30; i; i >>= 1)
	{
		if (i & 1)
		{
			// vector addition of matrix column by XOR
			r1 ^= v1; 
			r2 ^= v2 << 1;
		}
		// update matrix columns 
		v1 |= v1 >> 1; 
		v2 ^= v2 >> 1;
	}
	// map to unit cube [0,1)^3
	result[0] = r1 * (1.f / (1ULL << 32));
	result[1] = r2 * (1.f / (1ULL << 32));
	return result;
}

//------------------------------------------------------------------------

extern "C" __global__ void rayGenPrimaryKernel(void)
{
    // Get parameters.

    const RayGenPrimaryInput& in = *(const RayGenPrimaryInput*)c_RayGenPrimaryInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= in.w * in.h)
        return;

    // Compute end position.

    int pixel = ((const S32*)in.indexToPixel)[taskIdx];
    int posIdx = pixel;

    Vec4f nscreenPos;
    nscreenPos.x = 2.0f * ((F32)(posIdx % in.w) + 0.5f) / (F32)in.w - 1.0f;
    nscreenPos.y = 2.0f * ((F32)(posIdx / in.w) + 0.5f) / (F32)in.h - 1.0f;
    nscreenPos.z = 0.0f;
    nscreenPos.w = 1.0f;

    Vec4f worldPos4D = in.nscreenToWorld * nscreenPos;
    Vec3f worldPos = worldPos4D.getXYZ() / worldPos4D.w;

    // Write results.

    Ray& ray = ((Ray*)in.rays)[taskIdx];
    ((S32*)in.slotToID)[taskIdx] = pixel;
    ((S32*)in.idToSlot)[pixel] = taskIdx;

    ray.origin      = in.origin;
    ray.direction   = normalize(worldPos - in.origin);
    ray.tmin        = 0.0f;
    ray.tmax        = in.maxDist;
}

//------------------------------------------------------------------------

extern "C" __global__ void rayGenAOKernel(void)
{
    // Get parameters.

    const RayGenAOInput& in = c_RayGenAOInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= in.numInputRays)
        return;

    // Initialize.

    int                     inSlot      = taskIdx + in.firstInputSlot;
    const Ray&              inRay       = ((const Ray*)in.inRays)[inSlot];
    const RayResult&        inResult    = ((const RayResult*)in.inResults)[inSlot];
    int                     outSlot     = taskIdx * in.numSamples;
    Ray*                    outRays     = (Ray*)in.outRays + outSlot;
    S32*                    outIDToSlot = (S32*)in.outIDToSlot + outSlot;
    S32*                    outSlotToID = (S32*)in.outSlotToID + outSlot;
    const Vec3f*        normals     = (const Vec3f*)in.normals;

    // Compute origin, backtracking a little bit.

    F32 epsilon = 1.0e-4f;
    Vec3f origin = inRay.origin + inRay.direction * fmaxf(inResult.t - epsilon, 0.0f);

    // Lookup normal, flipping back-facing directions.

    int tri = inResult.id;
    Vec3f normal(1.0f, 0.0f, 0.0f);
    if (tri != -1)
        normal = normals[tri];
    if (dot(normal, inRay.direction) > 0.0f)
        normal = -normal;

    // Construct perpendicular vectors.

    Vec3f na = abs(normal);
    F32 nm = fmaxf(fmaxf(na.x, na.y), na.z);
    Vec3f perp(normal.y, -normal.x, 0.0f); // assume y is largest
    if (nm == na.z)
        perp = Vec3f(0.0f, normal.z, -normal.y);
    else if (nm == na.x)
        perp = Vec3f(-normal.z, 0.0f, normal.x);

    perp = normalize(perp);
    Vec3f biperp = cross(normal, perp);

    // Pick random rotation angle.

    U32 hashA = in.randomSeed + taskIdx;
    U32 hashB = 0x9e3779b9u;
    U32 hashC = 0x9e3779b9u;
    jenkinsMix(hashA, hashB, hashC);
    jenkinsMix(hashA, hashB, hashC);
    F32 angle = 2.0f * FW_PI * (F32)hashC * exp2(-32);

    // Construct rotated tangent vectors.

    Vec3f t0 = perp * cosf(angle) + biperp * sinf(angle);
    Vec3f t1 = perp * -sinf(angle) + biperp * cosf(angle);

    // Generate each sample.

    for (int i = 0; i < in.numSamples; i++)
    {
        // Base-2 Halton sequence for X.

        F32 x = 0.0f;
        F32 xadd = 1.0f;
        unsigned int hc2 = i + 1;
        while (hc2 != 0)
        {
            xadd *= 0.5f;
            if ((hc2 & 1) != 0)
                x += xadd;
            hc2 >>= 1;
        }

        // Base-3 Halton sequence for Y.

        F32 y = 0.0f;
        F32 yadd = 1.0f;
        int hc3 = i + 1;
        while (hc3 != 0)
        {
            yadd *= 1.0f / 3.0f;
            y += (F32)(hc3 % 3) * yadd;
            hc3 /= 3;
        }

        // Warp to a point on the unit hemisphere.

        F32 angle = 2.0f * FW_PI * y;
        F32 r = sqrtf(x);
        x = r * cosf(angle);
        y = r * sinf(angle);
        float z = sqrtf(1.0f - x * x - y * y);

        // Output ray.

        outRays[i].origin       = origin;
        outRays[i].direction    = normalize(x * t0 + y * t1 + z * normal);
        outRays[i].tmin         = 0.0f;
        outRays[i].tmax         = (tri == -1) ? -1.0f : in.maxDist;
        outIDToSlot[i]          = i + outSlot;
        outSlotToID[i]          = i + outSlot;
    }
}

//------------------------------------------------------------------------

extern "C" __global__ void rayGenShadowKernel(void)
{
    // Get parameters.

    const RayGenShadowInput& in = c_RayGenShadowInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= in.numInputRays)
        return;

    // Initialize.

    int                     inSlot      = taskIdx + in.firstInputSlot;
    const Ray&              inRay       = ((const Ray*)in.inRays)[inSlot];
    const RayResult&        inResult    = ((const RayResult*)in.inResults)[inSlot];
    int                     outSlot     = taskIdx * in.numSamples;
    Ray*                    outRays     = (Ray*)in.outRays + outSlot;
    S32*                    outIDToSlot = (S32*)in.outIDToSlot + outSlot;
    S32*                    outSlotToID = (S32*)in.outSlotToID + outSlot;

    // Compute origin, backtracking a little bit.

    F32 epsilon = 1.0e-4f;
    Vec3f origin = inRay.origin + inRay.direction * fmaxf(inResult.t - epsilon, 0.0f);

    // Pick random offset.

    U32 hashA = in.randomSeed + taskIdx;
    U32 hashB = 0x9e3779b9u;
    U32 hashC = 0x9e3779b9u;
    jenkinsMix(hashA, hashB, hashC);
    jenkinsMix(hashA, hashB, hashC);
	Vec3f offset((F32)hashA*exp2(-32),(F32)hashB*exp2(-32),(F32)hashC*exp2(-32));

    // Generate each sample.
	const Vec3f lightPosition(in.lightPositionX,in.lightPositionY,in.lightPositionZ);
    const int tri = inResult.id;

	for (int i = 0; i < in.numSamples; i++)
    {
		// QMC.

		Vec3f pos(sobol2D(i),hammersley(i,in.numSamples));		// [0,1]
		pos += offset;											// Cranley-Patterson
		if(pos[0]>=1.f)	pos[0] -= 1.f;
		if(pos[1]>=1.f)	pos[1] -= 1.f;
		if(pos[2]>=1.f)	pos[2] -= 1.f;
		pos = pos*2-1;											// [-1,1]

		// Target position.

        const Vec3f target    = lightPosition + in.lightRadius * pos;
        const Vec3f direction = target - origin;

		// Output ray.

		outRays[i].origin       = origin;
		outRays[i].direction    = direction.normalized();
		outRays[i].tmin         = 0.0f;
		outRays[i].tmax         = (tri == -1) ? -1.0f : direction.length();
		outIDToSlot[i]          = i + outSlot;
		outSlotToID[i]          = i + outSlot;
	}
}

//------------------------------------------------------------------------
