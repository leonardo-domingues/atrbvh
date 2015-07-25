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
#include "gui/Window.hpp"
#include "gui/CommonControls.hpp"
#include "3d/CameraControls.hpp"
#include "cuda/Renderer.hpp"

namespace FW
{
//------------------------------------------------------------------------

class App : public Window::Listener, public CommonControls::StateObject
{
private:
    enum Action
    {
        Action_None,

        Action_About,
        Action_LoadMesh,

        Action_ResetCamera,
        Action_ExportCameraSignature,
        Action_ImportCameraSignature,
    };

public:
                        App             (void);
    virtual             ~App            (void);

    virtual bool        handleEvent     (const Window::Event& ev);
    virtual void        readState       (StateDump& d);
    virtual void        writeState      (StateDump& d) const;

    void                setWindowSize   (const Vec2i& size)         { m_window.setSize(size); }
    bool                loadState       (const String& fileName)    { return m_commonCtrl.loadState(fileName); }
    void                loadDefaultState(void)                      { if (!m_commonCtrl.loadState(m_commonCtrl.getStateFileName(1))) firstTimeInit(); }
    void                flashButtonTitles(void)                     { m_commonCtrl.flashButtonTitles(); }

private:
    void                rebuildGui      (void);
    void                waitKey         (void);
    void                render          (GLContext* gl);
    void                renderGuiHelp   (GLContext* gl);

    bool                loadMesh        (const String& fileName);
    void                resetCamera     (void);
    void                firstTimeInit   (void);

private:
                        App             (const App&); // forbidden
    App&                operator=       (const App&); // forbidden

private:
    Window              m_window;
    CommonControls      m_commonCtrl;
    CameraControls      m_cameraCtrl;
    Renderer            m_renderer;

    Action              m_action;
    String              m_meshFileName;
    MeshBase*           m_mesh;
    Renderer::RayType   m_rayType;
    F32                 m_aoRadius;
    S32                 m_numSamples;
    Array<String>       m_kernelNames;
    S32                 m_kernelNameIdx;

    bool                m_showHelp;
    bool                m_showCameraControls;
    bool                m_showKernelSelector;
    bool                m_guiDirty;
};

//------------------------------------------------------------------------

void    listKernels     (Array<String>& kernelNames);
void    runInteractive  (const Vec2i& frameSize, const String& stateFile);

void runBenchmark(
    const Vec2i&            frameSize,
    const String&           meshFile,
    const Array<String>&    cameras,
    const Array<String>&    kernels,
    F32                     sbvhAlpha,
    F32                     aoRadius,
    int                     numSamples,
    bool                    sortSecondary,
    int                     warmupRepeats,
    int                     measureRepeats);

//------------------------------------------------------------------------
}
