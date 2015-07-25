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
#include "io/Stream.hpp"
#include "bvh/BVH.hpp"
#include "kernels/CudaTracerKernels.hpp"

#include "BVHTree.h"
#include "BVHCollapsedTree.h"
#include "Scene.h"

namespace FW
{
//------------------------------------------------------------------------
// Nodes / BVHLayout_Compact
//      nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[innerOfs + 48] = Vec4i(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, 0, 0)
//
// TriWoop / BVHLayout_Compact
//      triWoop[triOfs*16 + 0 ] = Vec4f(woopZ)
//      triWoop[triOfs*16 + 16] = Vec4f(woopU)
//      triWoop[triOfs*16 + 32] = Vec4f(woopV)
//      triWoop[endOfs*16 + 0 ] = Vec4f(-0.0f, -0.0f, -0.0f, -0.0f)
//
// TriIndex / BVHLayout_Compact
//      triIndex[triOfs*4] = origIdx
//
//------------------------------------------------------------------------
//
// Nodes / BVHLayout_AOS_AOS, BVHLayout_AOS_SOA
//      nodes[node*64  + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[node*64  + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[node*64  + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[inner*64 + 48] = Vec4f(c0.inner or ~c0.leaf, c1.inner or ~c1.leaf, 0, 0)
//      nodes[leaf*64  + 48] = Vec4i(triStart, triEnd, 0, 0)
//
// Nodes / BVHLayout_SOA_AOS, BVHLayout_SOA_SOA
//      nodes[node*16  + size*0/4] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[node*16  + size*1/4] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[node*16  + size*2/4] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[inner*16 + size*3/4] = Vec4f(c0.inner or ~c0.leaf, c1.inner or ~c1.leaf, 0, 0)
//      nodes[leaf*16  + size*3/4] = Vec4i(triStart, triEnd, 0, 0)
//
// TriWoop / BVHLayout_AOS_AOS, BVHLayout_SOA_AOS
//      triWoop[tri*64 + 0 ] = Vec4f(woopZ)
//      triWoop[tri*64 + 16] = Vec4f(woopU)
//      triWoop[tri*64 + 32] = Vec4f(woopV)
//
// TriWoop / BVHLayout_AOS_SOA, BVHLayout_SOA_SOA
//      triWoop[tri*16 + size*0/4] = Vec4f(woopZ)
//      triWoop[tri*16 + size*1/4] = Vec4f(woopU)
//      triWoop[tri*16 + size*2/4] = Vec4f(woopV)
//
// TriIndex / BVHLayout_AOS_AOS, BVHLayout_AOS_SOA, BVHLayout_SOA_AOS, BVHLayout_SOA_SOA
//      triIndex[tri*4] = origIdx
//------------------------------------------------------------------------

class CudaBVH
{
public:
    enum
    {
        Align = 4096
    };

public:
    explicit    CudaBVH             (const BVH& bvh, BVHLayout layout);
    CudaBVH             (CudaBVH& other)
    {
        operator=(other);
    }
    CudaBVH             (unsigned int numberOfTriangles, const BVHRT::BVHTree* tree, 
                        FW::Scene* scene, BVHLayout layout);
    CudaBVH             (const BVHRT::BVHCollapsedTree* tree, FW::Scene* scene, BVHLayout layout);
    explicit    CudaBVH             (InputStream& in);
    ~CudaBVH            (void);

    BVHLayout   getLayout           (void) const
    {
        return m_layout;
    }
    Buffer&     getNodeBuffer       (void)
    {
        return m_nodes;
    }
    Buffer&     getTriWoopBuffer    (void)
    {
        return m_triWoop;
    }
    Buffer&     getTriIndexBuffer   (void)
    {
        return m_triIndex;
    }

    // AOS: idx ignored, returns entire buffer
    // SOA: 0 <= idx < 4, returns one subarray
    Vec2i       getNodeSubArray     (int idx) const; // (ofs, size)
    Vec2i       getTriWoopSubArray  (int idx) const; // (ofs, size)

    CudaBVH&    operator=           (CudaBVH& other);

    void        serialize           (OutputStream& out);

private:
    void        createNodeBasic     (const BVH& bvh);
    void        createTriWoopBasic  (const BVH& bvh);
    void        createTriIndexBasic (const BVH& bvh);
    void        createCompact       (const BVH& bvh, int nodeOffsetSizeDiv);
    void        createCompact       (int numberOfTriangles, const BVHRT::BVHTree* tree,
                                    Scene* scene, int nodeOffsetSizeDiv);
    void        createCompact       (const BVHRT::BVHCollapsedTree* tree, Scene* scene, 
                                    int nodeOffsetSizeDiv);

    void        woopifyTri          (const BVH& bvh, int idx);
    void        woopifyTri          (Buffer& indexBuffer, Buffer& vertexBuffer, int dataIndex);

private:
    BVHLayout   m_layout;
    Buffer      m_nodes;
    Buffer      m_triWoop;
    Buffer      m_triIndex;
    Vec4f       m_woop[3];
};

//------------------------------------------------------------------------
}
