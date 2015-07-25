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

/*
    GF100-optimized variant of the "Speculative while-while"
    kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#include "CudaTracerKernels.hpp"

//------------------------------------------------------------------------

#define STACK_SIZE  64  // Size of the traversal stack in local memory.

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
    g_config.bvhLayout = BVHLayout_Compact;
    g_config.blockWidth = 32; // One warp per row.
    g_config.blockHeight = 4; // 4*32 = 128 threads, optimal for GTX480
}

//------------------------------------------------------------------------

TRACE_FUNC
{
    // Traversal stack in CUDA thread-local memory.

    int traversalStack[STACK_SIZE];

    // Live state during traversal, stored in registers.

    int     rayidx;                 // Ray index.
    float   origx, origy, origz;    // Ray origin.
    float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.
    float   idirx, idiry, idirz;    // 1 / dir
    float   oodx, oody, oodz;       // orig / dir

    char*   stackPtr;               // Current position in traversal stack.
    int     leafAddr;               // First postponed leaf, non-negative if none.
    int     nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.

    // Initialize.
    {
        // Pick ray index.

        rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
        if (rayidx >= numRays)
            return;

        // Fetch ray.

        float4 o = rays[rayidx * 2 + 0];
        float4 d = rays[rayidx * 2 + 1];
        origx = o.x, origy = o.y, origz = o.z;
        dirx = d.x, diry = d.y, dirz = d.z;
        tmin = o.w;

        float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        oodx = origx * idirx, oody = origy * idiry, oodz = origz * idirz;

        // Setup traversal.

        traversalStack[0] = EntrypointSentinel; // Bottom-most entry.
        stackPtr = (char*)&traversalStack[0];
        leafAddr = 0;   // No postponed leaf.
        nodeAddr = 0;   // Start from the root.
        hitIndex = -1;  // No triangle intersected so far.
        hitT     = d.w; // tmax
    }

    // Traversal loop.

    while (nodeAddr != EntrypointSentinel)
    {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        bool searchingLeaf = true;
        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
        {
            // Fetch AABBs of the two child nodes.

            float4* ptr = (float4*)((char*)nodesA + nodeAddr);
            float4 n0xy = ptr[0]; // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            float4 n1xy = ptr[1]; // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            float4 nz   = ptr[2]; // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)

            // Intersect the ray against the child nodes.

            float c0lox = n0xy.x * idirx - oodx;
            float c0hix = n0xy.y * idirx - oodx;
            float c0loy = n0xy.z * idiry - oody;
            float c0hiy = n0xy.w * idiry - oody;
            float c0loz = nz.x   * idirz - oodz;
            float c0hiz = nz.y   * idirz - oodz;
            float c1loz = nz.z   * idirz - oodz;
            float c1hiz = nz.w   * idirz - oodz;
			float c0min = spanBeginFermi(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
			float c0max = spanEndFermi  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
            float c1lox = n1xy.x * idirx - oodx;
            float c1hix = n1xy.y * idirx - oodx;
            float c1loy = n1xy.z * idiry - oody;
            float c1hiy = n1xy.w * idiry - oody;
			float c1min = spanBeginFermi(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
			float c1max = spanEndFermi  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

            bool traverseChild0 = (c0max >= c0min);
            bool traverseChild1 = (c1max >= c1min);

            // Neither child was intersected => pop stack.

            if (!traverseChild0 && !traverseChild1)
            {
                nodeAddr = *(int*)stackPtr;
                stackPtr -= 4;
            }

            // Otherwise => fetch child pointers.

            else
            {
                int2 cnodes = *(int2*)&ptr[3];
                nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                // Both children were intersected => push the farther one.

                if (traverseChild0 && traverseChild1)
                {
                    if (c1min < c0min)
                        swap(nodeAddr, cnodes.y);
                    stackPtr += 4;
                    *(int*)stackPtr = cnodes.y;
                }
            }

            // First leaf => postpone and continue traversal.

            if (nodeAddr < 0 && leafAddr >= 0)
            {
                searchingLeaf = false;
                leafAddr = nodeAddr;
                nodeAddr = *(int*)stackPtr;
                stackPtr -= 4;
            }

            // All SIMD lanes have found a leaf => process them.

            if (!__any(searchingLeaf))
                break;
        }

        // Process postponed leaf nodes.

        while (leafAddr < 0)
        {
            // Intersect the ray against each triangle using Sven Woop's algorithm.

            for (int triAddr = ~leafAddr;; triAddr += 3)
            {
                // Read first 16 bytes of the triangle.
                // End marker (negative zero) => all triangles processed.

                float4 v00 = tex1Dfetch(t_trisA, triAddr + 0);
                if (__float_as_int(v00.x) == 0x80000000)
                    break;

                // Compute and check intersection t-value.

                float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    float4 v11 = tex1Dfetch(t_trisA, triAddr + 1);
                    float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    float u = Ox + t*Dx;

                    if (u >= 0.0f && u <= 1.0f)
                    {
                        // Compute and check barycentric v.

                        float4 v22 = tex1Dfetch(t_trisA, triAddr + 2);
                        float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.

                            hitT = t;
                            hitIndex = triAddr;
                            if (anyHit)
                            {
                                nodeAddr = EntrypointSentinel;
                                break;
                            }
                        }
                    }
                }
            } // triangle

            // Another leaf was postponed => process it as well.

            leafAddr = nodeAddr;
            if(nodeAddr<0)
            {
                nodeAddr = *(int*)stackPtr;
                stackPtr -= 4;
            }
        } // leaf
    } // traversal

    // Remap intersected triangle index, and store the result.

    if (hitIndex != -1)
        hitIndex = tex1Dfetch(t_triIndices, hitIndex);
    STORE_RESULT(rayidx, hitIndex, hitT);
}

//------------------------------------------------------------------------
