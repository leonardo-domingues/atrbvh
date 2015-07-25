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

#include "ray/PixelTable.hpp"

using namespace FW;

//------------------------------------------------------------------------

PixelTable::PixelTable(void)
: m_size(0)
{
}

//------------------------------------------------------------------------

PixelTable::~PixelTable(void)
{
}

//------------------------------------------------------------------------

void PixelTable::setSize(const Vec2i& size)
{
    bool recalc = (size != m_size);
    m_size = size;
    if (recalc)
        recalculate();
}

//------------------------------------------------------------------------

void PixelTable::recalculate(void)
{
    // construct LUTs
    m_indexToPixel.resizeDiscard(m_size.x * m_size.y * sizeof(S32));
    m_pixelToIndex.resizeDiscard(m_size.x * m_size.y * sizeof(S32));
    S32* idxtopos = (S32*)m_indexToPixel.getMutablePtr();
    S32* postoidx = (S32*)m_pixelToIndex.getMutablePtr();

    // dumb mode
#if 0
    {
        for (int i=0; i < width*height; i++)
        {
            postoidx[i] = i;
            idxtopos[i] = i;
        }
        return;
    }
#endif

    // smart mode
    int idx     = 0;
    int bheight = m_size.y & ~7;
    int bwidth  = m_size.x  & ~7;

    // bulk of the image, sort blocks in in morton order
    int maxdim = (bwidth > bheight) ? bwidth : bheight;

    // round up to nearest power of two
    maxdim |= maxdim >> 1;
    maxdim |= maxdim >> 2;
    maxdim |= maxdim >> 4;
    maxdim |= maxdim >> 8;
    maxdim |= maxdim >> 16;
    maxdim = (maxdim + 1) >> 1;

    int width8  = bwidth >> 3;
    int height8 = bheight >> 3;
    for (int i=0; i < maxdim*maxdim; i++)
    {
        // get interleaved bit positions
        int tx = 0;
        int ty = 0;
        int val = i;
        int bit = 1;
        while (val)
        {
            if (val & 1) tx |= bit;
            if (val & 2) ty |= bit;
            bit += bit;
            val >>= 2;
        }
        if (tx < width8 && ty < height8)
        {
            for (int inner=0; inner < 64; inner++)
            {
                // swizzle ix and iy within blocks as well
                int ix = ((inner&1)>>0) | ((inner&4)>>1) | ((inner&16)>>2);
                int iy = ((inner&2)>>1) | ((inner&8)>>2) | ((inner&32)>>3);
                int pos = (ty*8 + iy) * m_size.x + (tx*8 + ix);

                postoidx[pos] = idx;
                idxtopos[idx++] = pos;
            }
        }
    }

    // if height not divisible, add horizontal stripe below bulk
    for (int px=0; px < bwidth; px++)
    for (int py=bheight; py < m_size.y; py++)
    {
        int pos = px + py * m_size.x;
        postoidx[pos] = idx;
        idxtopos[idx++] = pos;
    }

    // if width not divisible, add vertical stripe and the corner
    for (int py=0; py < m_size.y; py++)
    for (int px=bwidth; px < m_size.x; px++)
    {
        int pos = px + py * m_size.x;
        postoidx[pos] = idx;
        idxtopos[idx++] = pos;
    }

    // done!
}
