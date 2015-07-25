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
    "Persistent while-while kernel" used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#include "CudaTracerKernels.hpp"

//------------------------------------------------------------------------

#define NODES_ARRAY_OF_STRUCTURES           // Define for AOS, comment out for SOA.
#define TRIANGLES_ARRAY_OF_STRUCTURES       // Define for AOS, comment out for SOA.

#define LOAD_BALANCER_BATCH_SIZE        96  // Number of rays to fetch at a time. Must be a multiple of 32.
#define STACK_SIZE                      64  // Size of the traversal stack in local memory.
#define LOOP_NODE                       100 // Nodes: 1 = if, 100 = while.
#define LOOP_TRI                        100 // Triangles: 1 = if, 100 = while.

extern "C" __device__ int g_warpCounter;    // Work counter for persistent threads.

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
    // Temporary data stored in shared memory to reduce register pressure.

    __shared__ RayStruct shared[32 * MaxBlockHeight + 1];
    RayStruct* aux = shared + threadIdx.x + (blockDim.x * threadIdx.y);

    // Traversal stack in CUDA thread-local memory.
    // Allocate 3 additional entries for spilling rarely used variables.

    int traversalStack[STACK_SIZE + 3];
    traversalStack[STACK_SIZE + 0] = threadIdx.x; // Forced to local mem => saves a register.
    traversalStack[STACK_SIZE + 1] = threadIdx.y;
    // traversalStack[STACK_SIZE + 2] holds ray index.

    // Live state during traversal, stored in registers.

    float   origx, origy, origz;    // Ray origin.
    int     stackPtr;               // Current position in traversal stack.
    int     nodeAddr;               // Current internal node.
    int     triAddr;                // Start of a pending triangle list.
    int     triAddr2;               // End of a pending triangle list.
    float   hitT;                   // t-value of the closest intersection.

    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.
    __shared__ volatile int rayCountArray[MaxBlockHeight]; // Number of rays in the local pool.
    nextRayArray[threadIdx.y] = 0;
    rayCountArray[threadIdx.y] = 0;

    // Persistent threads: fetch and process rays in a loop.

    do
    {
        int tidx = traversalStack[STACK_SIZE + 0]; // threadIdx.x
        int widx = traversalStack[STACK_SIZE + 1]; // threadIdx.y
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
            int rayidx = localPoolNextRay + tidx;
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

            float ooeps = exp2f(-80.0f); // Avoid div by zero.
            aux->idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            aux->idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            aux->idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
            traversalStack[STACK_SIZE + 2] = rayidx; // Spill.

            // Setup traversal.

            traversalStack[0] = EntrypointSentinel; // Bottom-most entry.
            stackPtr = 0;
            nodeAddr = 0;   // Start from the root.
            triAddr  = 0;   // No pending triangle list.
            triAddr2 = 0;
            STORE_RESULT(rayidx, -1, 0.0f); // No triangle intersected so far.
            hitT     = d.w; // tmax
        }

        // Traversal loop.

        do
        {
            // Traverse internal nodes.

            for (int i = LOOP_NODE - 1; i >= 0 && nodeAddr >= 0 && nodeAddr != EntrypointSentinel; i--)
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

                // Decide where to go next.

                bool traverseChild0 = (c0max >= c0min);
                bool traverseChild1 = (c1max >= c1min);

                nodeAddr           = __float_as_int(cnodes.x);      // stored as int
                int nodeAddrChild1 = __float_as_int(cnodes.y);      // stored as int

                // One child was intersected => go there.

                if(traverseChild0 != traverseChild1)
                {
                    if (traverseChild1)
                        nodeAddr = nodeAddrChild1;
                }
                else
                {
                    // Neither child was intersected => pop.

                    if (!traverseChild0)
                    {
                        nodeAddr = traversalStack[stackPtr];
                        --stackPtr;
                    }

                    // Both children were intersected => push the farther one.

                    else
                    {
                        if(c1min < c0min)
                            swap(nodeAddr, nodeAddrChild1);
                        ++stackPtr;
                        traversalStack[stackPtr] = nodeAddrChild1;
                    }
                }
            }

            // Current node is a leaf => fetch the start and end of the triangle list.

            if (nodeAddr < 0 && triAddr >= triAddr2)
            {
#ifdef NODES_ARRAY_OF_STRUCTURES
                float4 leaf=FETCH_TEXTURE(nodesA, (-nodeAddr-1)*4+3, float4);
#else
                float4 leaf=FETCH_TEXTURE(nodesD, (-nodeAddr-1), float4);
#endif
                triAddr  = __float_as_int(leaf.x); // stored as int
                triAddr2 = __float_as_int(leaf.y); // stored as int

                // Pop.

                nodeAddr = traversalStack[stackPtr];
                --stackPtr;
            }

            // Intersect the ray against each triangle using Sven Woop's algorithm.

            for (int i = LOOP_TRI - 1; i >= 0 && triAddr < triAddr2; triAddr++, i--)
            {
                // Compute and check intersection t-value.

#ifdef TRIANGLES_ARRAY_OF_STRUCTURES
                float4 v00 = FETCH_GLOBAL(trisA, triAddr*4+0, float4);
                float4 v11 = FETCH_GLOBAL(trisA, triAddr*4+1, float4);
#else
                float4 v00 = FETCH_GLOBAL(trisA, triAddr, float4);
                float4 v11 = FETCH_GLOBAL(trisB, triAddr, float4);
#endif
                float dirx  = 1.0f / aux->idirx;
                float diry  = 1.0f / aux->idiry;
                float dirz  = 1.0f / aux->idirz;

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
                            STORE_RESULT(traversalStack[STACK_SIZE + 2], FETCH_GLOBAL(triIndices, triAddr, int), t);
                            if (anyHit)
                            {
                                nodeAddr = EntrypointSentinel;
                                triAddr = triAddr2; // Breaks the do-while.
                                break;
                            }
                        }
                    }
                }
            } // triangle
        } while (nodeAddr != EntrypointSentinel || triAddr < triAddr2); // traversal
    } while(aux); // persistent threads (always true)
}

//------------------------------------------------------------------------
