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
    "Persistent packet traversal" kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009

    Adds persistent threads to the algorithm explained in:

    "Realtime ray tracing on GPU with BVH-based packet",
    Johannes Guenther, Stefan Popov, Hans-Peter Seidel and Philipp Slusallek,
    Proc. IEEE/Eurographics Symposium on Interactive Ray Tracing 2007, 113–118.
*/

#include "CudaTracerKernels.hpp"

//------------------------------------------------------------------------

#define NODES_ARRAY_OF_STRUCTURES               // Define for AOS, comment out for SOA.
#define TRIANGLES_ARRAY_OF_STRUCTURES           // Define for AOS, comment out for SOA.

#define LOAD_BALANCER_BATCH_SIZE        96      // Number of rays to fetch at a time. Must be a multiple of 32.
#define STACK_SIZE                      (23+32) // Size of the traversal stack in shared memory.

extern "C" __device__ int g_warpCounter;        // Work counter for persistent threads.

//------------------------------------------------------------------------

__device__ void reduceSum(int* red, int tidx) // Warp-wide integer sum.
{
    red[tidx] += red[tidx ^ 1];
    red[tidx] += red[tidx ^ 2];
    red[tidx] += red[tidx ^ 4];
    red[tidx] += red[tidx ^ 8];
    red[tidx] += red[tidx ^ 16];
}

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
#if (defined(NODES_ARRAY_OF_STRUCTURES) && defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_AOS_AOS;
#elif (defined(NODES_ARRAY_OF_STRUCTURES) && !defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_AOS_SOA;
#elif (!defined(NODES_ARRAY_OF_STRUCTURES) && defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_SOA_AOS;
#elif (!defined(NODES_ARRAY_OF_STRUCTURES) && !defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_SOA_SOA;
#endif

    g_config.blockWidth = 32; // One warp per row.
    g_config.blockHeight = 6; // 6*32 = 192 threads, optimal for GTX285.
    g_config.usePersistentThreads = 1;
}

//------------------------------------------------------------------------

TRACE_FUNC
{
    // Shared memory arrays.

    __shared__ RayStruct shared[32 * MaxBlockHeight + 1];
    __shared__ volatile int s_stack[STACK_SIZE * MaxBlockHeight];
    __shared__ volatile int s_stackPtr[MaxBlockHeight]; // NOTE: This could equally well be in a register.
    __shared__ volatile int s_red[32 * MaxBlockHeight];

    RayStruct*    aux      = shared + threadIdx.x + (blockDim.x * threadIdx.y);
    volatile int* stack    = s_stack + STACK_SIZE * threadIdx.y;
    volatile int* red      = s_red + 32 * threadIdx.y;
    volatile int& stackPtr = s_stackPtr[threadIdx.y];

    // Live state during traversal, stored in registers.

    int     tidx = threadIdx.x;     // Lane index within warp.
    int     widx = threadIdx.y;     // Warp index within block.

    int     rayidx;                 // Ray index.
    float   origx, origy, origz;    // Ray origin.
    bool    valid;                  // False if the ray is degenerate.

    int     nodeAddr;               // Current node, negative if leaf.
    bool    terminated;             // Whether the traversal has been terminated.
    float   hitT;                   // t-value of the closest intersection.

    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.
    __shared__ volatile int rayCountArray[MaxBlockHeight]; // Number of rays in the local pool.
    nextRayArray[threadIdx.y] = 0;
    rayCountArray[threadIdx.y] = 0;

    // Persistent threads: fetch and process rays in a loop.

    do
    {
        volatile int& localPoolRayCount = rayCountArray[widx];
        volatile int& localPoolNextRay = nextRayArray[widx];

        // Local pool is empty => fetch new rays from the global pool using lane 0.

        if (tidx == 0 && localPoolRayCount <= 0)
        {
            localPoolNextRay = atomicAdd(&g_warpCounter, LOAD_BALANCER_BATCH_SIZE);
            localPoolRayCount = LOAD_BALANCER_BATCH_SIZE;
        }

        // Pick 32 rays from the local pool.
        // Out of work => done.
        {
            rayidx = localPoolNextRay + tidx;
            if (rayidx >= numRays)
                break;

            if (tidx == 0)
            {
                localPoolNextRay += 32;
                localPoolRayCount -= 32;
            }

            // Fetch ray.

            float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
            float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
            origx = o.x, origy = o.y, origz = o.z;
            aux->tmin = o.w;
            valid = (o.w < d.w);

            float ooeps = exp2f(-80.0f); // Avoid div by zero.
            aux->idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            aux->idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            aux->idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));

            // Setup traversal.

            stackPtr = -1;                  // Stack is empty.
            nodeAddr = 0;                   // Start from the root.
            terminated = false;             // Not terminated yet.
            STORE_RESULT(rayidx, -1, 0.0f); // No triangle intersected so far.
            hitT = d.w;                     // tmax
        }

        // Traversal loop.

        while (valid)
        {
            // Internal node => intersect children.

            if (nodeAddr >= 0)
            {
                // Fetch AABBs of the two child nodes.

#ifdef NODES_ARRAY_OF_STRUCTURES
                float4 n0xy = FETCH_TEXTURE(nodesA, nodeAddr*4+0, float4);  // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
                float4 n1xy = FETCH_TEXTURE(nodesA, nodeAddr*4+1, float4);  // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
                float4 nz   = FETCH_TEXTURE(nodesA, nodeAddr*4+2, float4);  // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
                float4 cnodes=FETCH_TEXTURE(nodesA, nodeAddr*4+3, float4);
#else
                float4 n0xy = FETCH_TEXTURE(nodesA, nodeAddr, float4);      // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
                float4 n1xy = FETCH_TEXTURE(nodesB, nodeAddr, float4);      // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
                float4 nz   = FETCH_TEXTURE(nodesC, nodeAddr, float4);      // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
                float4 cnodes=FETCH_TEXTURE(nodesD, nodeAddr, float4);
#endif

                // Intersect the ray against the child nodes.

                float oodx  = origx * aux->idirx;
                float oody  = origy * aux->idiry;
                float oodz  = origz * aux->idirz;
                float c0lox = n0xy.x * aux->idirx - oodx;
                float c0hix = n0xy.y * aux->idirx - oodx;
                float c0loy = n0xy.z * aux->idiry - oody;
                float c0hiy = n0xy.w * aux->idiry - oody;
                float c0loz = nz.x   * aux->idirz - oodz;
                float c0hiz = nz.y   * aux->idirz - oodz;
                float c1loz = nz.z   * aux->idirz - oodz;
                float c1hiz = nz.w   * aux->idirz - oodz;
                float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), aux->tmin);
                float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
                float c1lox = n1xy.x * aux->idirx - oodx;
                float c1hix = n1xy.y * aux->idirx - oodx;
                float c1loy = n1xy.z * aux->idiry - oody;
                float c1hiy = n1xy.w * aux->idiry - oody;
                float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), aux->tmin);
                float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

                // Perform warp-wide vote to decide where to go.

                bool traverseChild0 = (c0max >= c0min);
                bool traverseChild1 = (c1max >= c1min);
                bool anyc0 = __any(traverseChild0);
                bool anyc1 = __any(traverseChild1);
                int nodeAddrChild0 = __float_as_int(cnodes.x); // Stored as int.
                int nodeAddrChild1 = __float_as_int(cnodes.y); // Stored as int.

                // Both children were intersected => vote which one to visit first.

                if (anyc0 && anyc1)
                {
                    red[tidx] = (c1min < c0min) ? 1 : -1;
                    reduceSum((int*)red, tidx);
                    if (red[tidx] >= 0)
                        swap(nodeAddrChild0, nodeAddrChild1);

                    nodeAddr = nodeAddrChild0;
                    if (tidx == 0)
                    {
                        stackPtr++;
                        stack[stackPtr] = nodeAddrChild1; // Lane 0 writes.
                    }
                }

                // Only one child was intersected => go there.

                else if (anyc0)
                {
                    nodeAddr = nodeAddrChild0;
                }
                else if (anyc1)
                {
                    nodeAddr = nodeAddrChild1;
                }

                // Neither child was intersected => pop.

                else
                {
                    if (stackPtr < 0)
                        break;
                    else
                    {
                        nodeAddr = stack[stackPtr]; // All lanes read.
                        if (tidx == 0)
                            stackPtr--; // Lane 0 decrements.
                    }
                }
            } // internal node

            // Leaf node => intersect triangles.

            if (nodeAddr < 0)
            {
                // Fetch the start and end of the triangle list.

                nodeAddr = -nodeAddr-1;
#ifdef NODES_ARRAY_OF_STRUCTURES
                float4 leaf = FETCH_TEXTURE(nodesA, nodeAddr*4+3, float4);
#else
                float4 leaf = FETCH_TEXTURE(nodesD, nodeAddr, float4);
#endif
                int triAddr  = __float_as_int(leaf.x); // Stored as int.
                int triAddr2 = __float_as_int(leaf.y); // Stored as int.

                // Intersect the ray against each triangle using Sven Woop's algorithm.

                for(; triAddr < triAddr2; triAddr++)
                {
                    // Compute and check intersection t-value.

#ifdef TRIANGLES_ARRAY_OF_STRUCTURES
                    float4 v00 = FETCH_GLOBAL(trisA, triAddr*4+0, float4);
                    float4 v11 = FETCH_GLOBAL(trisA, triAddr*4+1, float4);
#else
                    float4 v00 = FETCH_GLOBAL(trisA, triAddr, float4);
                    float4 v11 = FETCH_GLOBAL(trisB, triAddr, float4);
#endif
                    float dirx = 1.0f / aux->idirx;
                    float diry = 1.0f / aux->idiry;
                    float dirz = 1.0f / aux->idirz;

                    float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                    float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                    float t = Oz * invDz;

                    if (t > aux->tmin && t < hitT)
                    {
                        // Compute and check barycentric u.

                        float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                        float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                        float u = Ox + t*Dx;

                        if (u >= 0.0f)
                        {
                            // Compute and check barycentric v.

#ifdef TRIANGLES_ARRAY_OF_STRUCTURES
                            float4 v22 = FETCH_GLOBAL(trisA, triAddr*4+2, float4);
#else
                            float4 v22 = FETCH_GLOBAL(trisC, triAddr, float4);
#endif
                            float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                            float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                            float v = Oy + t*Dy;

                            if (v >= 0.0f && u + v <= 1.0f)
                            {
                                // Record intersection.
                                // Closest intersection not required => terminate.

                                hitT = t;
                                STORE_RESULT(rayidx, FETCH_GLOBAL(triIndices, triAddr, int), t);
                                if (anyHit)
                                    terminated = true; // NOTE: Cannot break because packet traversal!
                            }
                        }
                    }
                } // triangle

                // All lanes have terminated => traversal done.

                if (__all(terminated))
                    break;

                // Pop stack.

                if (stackPtr < 0)
                    break;
                else
                {
                    nodeAddr = stack[stackPtr]; // Everyone reads.
                    if (tidx == 0)
                        stackPtr--; // Lane 0 decrements.
                }
            } // leaf node
        } // traversal loop
    } while(aux); // persistent threads (always true)
}

//------------------------------------------------------------------------
