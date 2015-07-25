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

#include "App.hpp"
#include "base/Main.hpp"
#include "base/Random.hpp"
#include "gpu/GLContext.hpp"
#include "3d/Mesh.hpp"
#include "io/StateDump.hpp"

#include <stdio.h>
#include <conio.h>

using namespace FW;

//------------------------------------------------------------------------

static const char* const s_kernelDir        = "src/rt/kernels";
static const char* const s_initialMeshDir   = "scenes/rt/fairyforest";
static const char* const s_defaultMeshFile  = "scenes/rt/fairyforest/fairyforest.obj";

//------------------------------------------------------------------------

static const char* const s_defaultCameras[] =
{
    "conference",   "6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100",
    "fairyforest",  "cIxMx/sK/Ty/EFu3z/5m9mWx/YPA5z/8///m007toC10AnAHx///Uy200",
    "sibenik",      "ytIa02G35kz1i:ZZ/0//iSay/5W6Ex19///c/05frY109Qx7w////m100",
    "sanmiguel",    "Yciwz1oRQmz/Xvsm005CwjHx/b70nx18tVI7005frY108Y/:x/v3/z100",
    NULL
};

//------------------------------------------------------------------------

static const char* const s_rayTypeNames[] =
{
    "primary",
    "AO",
    "diffuse"
};

//------------------------------------------------------------------------

static const char* const s_aboutText =
    "\"Understanding the Efficiency of Ray Traversal on GPUs\",\n"
    "Timo Aila and Samuli Laine,\n"
    "Proc. High-Performance Graphics 2009\n"
    "\n"
    "Implementation by Tero Karras, Timo Aila, and Samuli Laine\n"
    "Copyright 2009-2012 NVIDIA Corporation\n"
    "\n"
    "http://code.google.com/p/understanding-the-efficiency-of-ray-traversal-on-gpus/\n"
;

//------------------------------------------------------------------------

static const char* const s_commandHelpText =
    "\n"
    "Usage: rt <mode> [options]\n"
    "\n"
    "Supported values for <mode>:\n"
    "\n"
    "   interactive             Start in interactive mode.\n"
    "   benchmark               Run benchmark for given mesh.\n"
    "\n"
    "Common options:\n"
    "\n"
    "   --log=<file.log>        Log all output to file.\n"
    "   --size=<w>x<h>          Frame size. Default is \"1024x768\".\n"
    "\n"
    "Options for \"rt interactive\":\n"
    "\n"
    "   --state=<file.dat>      Load state from the given file.\n"
    "\n"
    "Options for \"rt benchmark\":\n"
    "\n"
    "   --mesh=<file.obj>       Mesh to benchmark.\n"
    "   --camera=\"<sig>\"        Camera signature. Can specify multiple times.\n"
    "   --kernel=<name>         CUDA kernel. Can specify multiple. Default = all.\n"
    "   --sbvh-alpha=<value>    SBVH alpha parameter. Default is \"1.0e-5\".\n"
    "   --ao-radius=<value>     AO ray length. Default is \"5\".\n"
    "   --samples=<value>       Secondary rays per pixel. Default is \"32\".\n"
    "   --sort=<1/0>            Sort secondary rays. Default is \"1\".\n"
    "   --warmup-repeats=<num>  Launches prior to measurement. Default is \"2\".\n"
    "   --measure-repeats=<num> Launches to measure per batch. Default is \"10\".\n"
    "\n"
;

//------------------------------------------------------------------------

static const char* const s_guiHelpText =
    "General keys:\n"
    "\n"
    "\tF1\tHide this message\n"
    "\tEsc\tExit (also Alt-F4)\n"
    "\tTab\tShow all GUI controls\n"
    "\tNum\tLoad numbered state\n"
    "\tAlt-Num\tSave numbered state\n"
    "\tF9\tShow/hide FPS counter\n"
    "\tF10\tShow/hide GUI\n"
    "\tF11\tToggle fullscreen mode\n"
    "\tPrtScn\tSave screenshot\n"
    "\n"
    "Camera movement:\n"
    "\n"
    "\tDrag\tRotate (left), strafe (middle), zoom (right)\n"
    "\tArrows\tRotate\n"
    "\tW\tMove forward (also Alt-UpArrow)\n"
    "\tS\tMove back (also Alt-DownArrow)\n"
    "\tA\tStrafe left (also Alt-LeftArrow)\n"
    "\tD\tStrafe right (also Alt-RightArrow)\n"
    "\tR\tStrafe up (also PageUp)\n"
    "\tF\tStrafe down (also PageDown)\n"
    "\tWheel\tAdjust movement speed\n"
    "\tSpace\tMove faster (hold)\n"
    "\tCtrl\tMove slower (hold)\n"
    "\n"
    "Uncheck \"Retain camera alignment\" to enable:\n"
    "\n"
    "\tQ\tRoll counter-clockwise (also Insert)\n"
    "\tE\tRoll clockwise (also Home)\n"
;

//------------------------------------------------------------------------

App::App(void)
:   m_commonCtrl            (CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5),
    m_cameraCtrl            (&m_commonCtrl, CameraControls::Feature_Default & ~CameraControls::Feature_NearSlider),
    m_action                (Action_None),
    m_mesh                  (NULL),

    m_rayType               (Renderer::RayType_Primary),
    m_aoRadius              (1.0f),
    m_numSamples            (4),
    m_kernelNameIdx         (0),

    m_showHelp              (false),
    m_showCameraControls    (false),
    m_showKernelSelector    (false),
    m_guiDirty              (false)
{
    listKernels(m_kernelNames);
    if (!m_kernelNames.getSize())
        fail("No CUDA kernel sources found!");

    m_commonCtrl.showFPS(true);
    m_commonCtrl.addStateObject(this);
    m_commonCtrl.setStateFilePrefix("state_rt_");

    m_window.setTitle("GPU Ray Traversal");
    m_window.addListener(&m_cameraCtrl);
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);

    rebuildGui();
}

//------------------------------------------------------------------------

App::~App(void)
{
    delete m_mesh;
}

//------------------------------------------------------------------------

bool App::handleEvent(const Window::Event& ev)
{
    // Window closed => destroy app.

    if (ev.type == Window::EventType_Close)
    {
        printf("Exiting...\n");
        m_window.showModalMessage("Exiting...");
        delete this;
        return true;
    }

    // Update GUI controls.

    if (ev.type == Window::EventType_KeyDown && ev.key == FW_KEY_TAB)
    {
        bool v = (!m_showCameraControls || !m_showKernelSelector);
        m_showCameraControls = v;
        m_showKernelSelector = v;
        m_guiDirty = true;
    }

    if (m_guiDirty)
    {
        m_guiDirty = false;
        rebuildGui();
    }

    // Handle actions.

    Action action = m_action;
    m_action = Action_None;
    String name;

    switch (action)
    {
    case Action_None:
        break;

    case Action_About:
        m_window.showMessageDialog("About", s_aboutText);
        break;

    case Action_LoadMesh:
        name = m_window.showFileLoadDialog("Load mesh", getMeshImportFilter(), s_initialMeshDir);
        if (name.getLength() && loadMesh(name))
            resetCamera();
        break;

    case Action_ResetCamera:
        if (m_mesh)
        {
            resetCamera();
            m_commonCtrl.message("Camera reset");
        }
        break;

    case Action_ExportCameraSignature:
        m_window.setVisible(false);
        printf("\nCamera signature:\n");
        printf("%s\n", m_cameraCtrl.encodeSignature().getPtr());
        waitKey();
        break;

    case Action_ImportCameraSignature:
        {
            m_window.setVisible(false);
            printf("\nEnter camera signature:\n");

            char buf[1024];
            if (fgets(buf, FW_ARRAY_SIZE(buf), stdin) != NULL)
                m_cameraCtrl.decodeSignature(buf);
            else
                setError("Signature too long!");

            if (!hasError())
                printf("Done.\n\n");
            else
            {
                printf("Error: %s\n", getError().getPtr());
                clearError();
                waitKey();
            }
        }
        break;

    default:
        FW_ASSERT(false);
        break;
    }

    // Repaint.

    m_window.setVisible(true);
    if (ev.type == Window::EventType_Paint)
        render(m_window.getGL());
    m_window.repaint();
    return false;
}

//------------------------------------------------------------------------

void App::readState(StateDump& s)
{
    String meshFileName;
    String kernelName;

    s.pushOwner("App");
    s.get(meshFileName,     "m_meshFileName");
    s.get((S32&)m_rayType,  "m_rayType");
    s.get(m_aoRadius,       "m_aoRadius");
    s.get(m_numSamples,     "m_numSamples");
    s.get(kernelName,       "m_kernelName");
    s.popOwner();

    if (m_meshFileName != meshFileName && meshFileName.getLength())
        loadMesh(meshFileName);
    if (m_kernelNames.contains(kernelName))
        m_kernelNameIdx = m_kernelNames.indexOf(kernelName);
    rebuildGui();
}

//------------------------------------------------------------------------

void App::writeState(StateDump& s) const
{
    s.pushOwner("App");
    s.set(m_meshFileName,                   "m_meshFileName");
    s.set((S32&)m_rayType,                  "m_rayType");
    s.set(m_aoRadius,                       "m_aoRadius");
    s.set(m_numSamples,                     "m_numSamples");
    s.set(m_kernelNames[m_kernelNameIdx],   "m_kernelName");
    s.popOwner();
}

//------------------------------------------------------------------------

void App::rebuildGui(void)
{
    CommonControls& cc = m_commonCtrl;
    cc.resetControls();

    cc.setControlVisibility(true);
    cc.addToggle(&m_showHelp,                                   FW_KEY_F1,      "Show help [F1]");
    cc.addButton((S32*)&m_action, Action_About,                 FW_KEY_NONE,    "About...");
    cc.addButton((S32*)&m_action, Action_LoadMesh,              FW_KEY_M,       "Load mesh... [M]");
    cc.addSeparator();

    cc.addToggle((S32*)&m_rayType, Renderer::RayType_Primary,   FW_KEY_F2,      "Trace primary rays [F2]", &m_guiDirty);
    cc.addToggle((S32*)&m_rayType, Renderer::RayType_AO,        FW_KEY_F3,      "Trace ambient occlusion rays [F3]", &m_guiDirty);
    cc.addToggle((S32*)&m_rayType, Renderer::RayType_Diffuse,   FW_KEY_F4,      "Trace diffuse rays [F4]", &m_guiDirty);
    cc.addSeparator();

    cc.beginSliderStack();
    cc.setControlVisibility(m_rayType == Renderer::RayType_AO);
    cc.addSlider(&m_aoRadius, 1.0e-3f, 1.0e4f, true, FW_KEY_NONE, FW_KEY_NONE,  "AO ray length = %g units");
    cc.setControlVisibility(m_rayType != Renderer::RayType_Primary);
    cc.addSlider(&m_numSamples, 1, 64, false, FW_KEY_NONE, FW_KEY_NONE,         "Secondary rays per pixel = %d");
    cc.endSliderStack();

    cc.setControlVisibility(m_showCameraControls);
    cc.addButton((S32*)&m_action, Action_ResetCamera,           FW_KEY_NONE,    "Reset camera");
    cc.addButton((S32*)&m_action, Action_ExportCameraSignature, FW_KEY_NONE,    "Export camera signature...");
    cc.addButton((S32*)&m_action, Action_ImportCameraSignature, FW_KEY_NONE,    "Import camera signature...");
    m_cameraCtrl.addGUIControls();
    cc.addSeparator();

    cc.setControlVisibility(m_showKernelSelector);
    for (int i = 0; i < m_kernelNames.getSize(); i++)
        cc.addToggle(&m_kernelNameIdx, i, FW_KEY_NONE, m_kernelNames[i]);
    cc.addSeparator();

    cc.setControlVisibility(true);
    cc.addToggle(&m_showCameraControls,                         FW_KEY_NONE,    "Show camera controls", &m_guiDirty);
    cc.addToggle(&m_showKernelSelector,                         FW_KEY_NONE,    "Show kernel selector", &m_guiDirty);
}

//------------------------------------------------------------------------

void App::waitKey(void)
{
    printf("Press any key to continue . . . ");
    _getch();
    printf("\n\n");
}

//------------------------------------------------------------------------

void App::render(GLContext* gl)
{
    // No mesh => display message.

    if (!m_mesh)
    {
        gl->drawModalMessage("No mesh loaded!");
        return;
    }

    // Set parameters.

    Renderer::Params params;
    params.kernelName   = m_kernelNames[m_kernelNameIdx];
    params.rayType      = m_rayType;
    params.aoRadius     = m_aoRadius;
    params.numSamples   = m_numSamples;

    m_renderer.setParams(params);
    m_renderer.setMessageWindow(&m_window);
    m_renderer.setEnableRandom(true);

    // Render.

    m_renderer.setMesh(m_mesh);
    F32 launchTime = m_renderer.renderFrame(gl, m_cameraCtrl);
    int numRays = m_renderer.getTotalNumRays();

    // Show statistics.

    CudaBVH* bvh = m_renderer.getCudaBVH();
    S64 nodeBytes = bvh->getNodeBuffer().getSize();
    S64 triBytes = bvh->getTriWoopBuffer().getSize() + bvh->getTriIndexBuffer().getSize();

    String rayStats = sprintf("%.2f million %s rays, %.2f ms, %.2f MRays/s",
        (F32)numRays * 1.0e-6f,
        s_rayTypeNames[m_rayType],
        launchTime * 1.0e3f,
        (F32)numRays * 1.0e-6f / launchTime);

    String bvhStats = sprintf("%.2f Mtris, %.2f MB (%.2f MB for nodes, %.2f MB for tris)",
        (F32)m_mesh->numTriangles() * 1.0e-6f,
        (F32)(nodeBytes + triBytes) * exp2(-20),
        (F32)nodeBytes * exp2(-20),
        (F32)triBytes * exp2(-20));

    m_commonCtrl.message(rayStats, "rayStats");
    m_commonCtrl.message(bvhStats, "bvhStats");

    // Show help.

    if (m_showHelp)
        renderGuiHelp(gl);
}

//------------------------------------------------------------------------

void App::renderGuiHelp(GLContext* gl)
{
    S32 fontSize = 16;
    F32 tabSize = 64.0f;

    Mat4f oldXform = gl->setVGXform(gl->xformMatchPixels());
    gl->setFont("Arial", fontSize, GLContext::FontStyle_Bold);
    Vec2f origin = Vec2f(8.0f, (F32)gl->getViewSize().y - 4.0f);

    String str = s_guiHelpText;
    int startIdx = 0;
    Vec2f pos = 0.0f;
    while (startIdx < str.getLength())
    {
        if (str[startIdx] == '\n')
            pos = Vec2f(0.0f, pos.y - (F32)fontSize);
        else if (str[startIdx] == '\t')
            pos.x += tabSize;

        int endIdx = startIdx;
        while (endIdx < str.getLength() && str[endIdx] != '\n' && str[endIdx] != '\t')
            endIdx++;

        gl->drawLabel(str.substring(startIdx, endIdx), pos + origin, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
        startIdx = max(endIdx, startIdx + 1);
    }

    gl->setVGXform(oldXform);
    gl->setDefaultFont();
}

//------------------------------------------------------------------------

bool App::loadMesh(const String& fileName)
{
    m_window.showModalMessage(sprintf("Loading mesh from '%s'...\nThis will take a few seconds.", fileName.getFileName().getPtr()));

    String oldError = clearError();
    MeshBase* mesh = importMesh(fileName);
    String newError = getError();

    if (restoreError(oldError))
    {
        delete mesh;
        String msg = sprintf("Error while loading '%s': %s", fileName.getPtr(), newError.getPtr());
        printf("%s\n", msg.getPtr());
        m_commonCtrl.message(msg);
        return false;
    }

    m_renderer.setMesh(NULL);
    delete m_mesh;
    m_meshFileName = fileName;
    m_mesh = mesh;
    m_commonCtrl.message(sprintf("Loaded mesh from '%s'", fileName.getPtr()));
    return true;
}

//------------------------------------------------------------------------

void App::resetCamera(void)
{
    if (!m_mesh)
        return;

    // Extract mesh name.

    String name = m_meshFileName;
    int slashIdx = max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
    int dotIdx = name.lastIndexOf('.');
    name = name.substring(slashIdx + 1, (dotIdx > slashIdx) ? dotIdx : name.getLength());

    // Look up default camera.

    for (const char* const* ptr = s_defaultCameras; ptr[0]; ptr += 2)
    {
        if (name == ptr[0])
        {
            m_cameraCtrl.decodeSignature(ptr[1]);
            return;
        }
    }

    // Not found => initialize based on mesh bounds.

    m_cameraCtrl.initForMesh(m_mesh);
}

//------------------------------------------------------------------------

void App::firstTimeInit(void)
{
    // Choose default kernel.

    String kernel = "tesla_persistent_while_while";
    if (CudaModule::isAvailable() && CudaModule::getComputeCapability() >= 12)
        kernel = "tesla_persistent_speculative_while_while";
    if (CudaModule::isAvailable() && CudaModule::getComputeCapability() >= 20)
        kernel = "fermi_speculative_while_while";
    if (CudaModule::isAvailable() && CudaModule::getComputeCapability() >= 30)
        kernel = "kepler_dynamic_fetch";
    if (m_kernelNames.contains(kernel))
        m_kernelNameIdx = m_kernelNames.indexOf(kernel);

    // Load mesh.

    loadMesh(s_defaultMeshFile);
    resetCamera();

    // Save state.

    m_commonCtrl.saveState(m_commonCtrl.getStateFileName(1));
    failIfError();
}

//------------------------------------------------------------------------

void FW::listKernels(Array<String>& kernelNames)
{
    WIN32_FIND_DATA fd;
    HANDLE h = FindFirstFile(sprintf("%s/*.cu", s_kernelDir).getPtr(), &fd);
    if (h != INVALID_HANDLE_VALUE)
    {
        do
        {
            String name = fd.cFileName;
            kernelNames.add(name.substring(0, name.getLength() - 3));
        }
        while (FindNextFile(h, &fd) != 0);
        FindClose(h);
    }
}

//------------------------------------------------------------------------

void FW::runInteractive(const Vec2i& frameSize, const String& stateFile)
{
    if (hasError())
        return;

    // Launch.

    printf("Starting up...\n");
    App* app = new App;
    app->setWindowSize(frameSize);

    // Load state.

    if (!hasError() && !stateFile.getLength())
        app->loadDefaultState();
    else if (!hasError() && !app->loadState(stateFile))
        setError("Unable to load state from '%s'!", stateFile.getPtr());

    // Error => close window.

    if (hasError())
        delete app;
    else
        app->flashButtonTitles();
}

//------------------------------------------------------------------------

void FW::runBenchmark(
    const Vec2i&            frameSize,
    const String&           meshFile,
    const Array<String>&    cameras,
    const Array<String>&    kernels,
    F32                     sbvhAlpha,
    F32                     aoRadius,
    int                     numSamples,
    bool                    sortSecondary,
    int                     warmupRepeats,
    int                     measureRepeats)
{
    int numRayTypes = Renderer::RayType_Max;

    // Print header.

    CudaModule::staticInit();
    printf("Running benchmark for \"%s\".\n", meshFile.getPtr());
    printf("\n");

    // Setup renderer.

    Renderer::Params params;
    params.aoRadius = aoRadius;
    params.numSamples = numSamples;
    params.sortSecondary = sortSecondary;

    BVH::BuildParams buildParams;
    buildParams.splitAlpha = sbvhAlpha;

    Renderer renderer;
    renderer.setBuildParams(buildParams);
    renderer.setMesh(importMesh(meshFile));

    // Create window.

    Window window;
    window.setSize(frameSize);
    window.setVisible(false);
    window.realize();
    GLContext* gl = window.getGL();

    // Error => skip.

    if (hasError())
        return;

    // Benchmark each combination.

    Array<F32> results;
    for (int kernelIdx = 0; kernelIdx < kernels.getSize(); kernelIdx++)
    {
        for (int rayType = 0; rayType < numRayTypes; rayType++)
        {
            S64 totalRays = 0;
            F32 totalLaunchTime = 0.0f;

            for (int cameraIdx = 0; cameraIdx < cameras.getSize(); cameraIdx++)
            {
                // Print status.

                String title = sprintf("%s, %s, camera %d", kernels[kernelIdx].getPtr(), s_rayTypeNames[rayType], cameraIdx);
                printf("%s...\n", title.getPtr());
                window.setTitle(title);

                // Setup rendering.

                params.kernelName = kernels[kernelIdx];
                params.rayType = (Renderer::RayType)rayType;
                renderer.setParams(params);

                CameraControls camera;
                camera.decodeSignature(cameras[cameraIdx]);
                renderer.beginFrame(gl, camera);
                totalRays += (S64)renderer.getTotalNumRays() * measureRepeats;

                // Process each batch.

                while (renderer.nextBatch())
                {
                    // Render and display result.

                    renderer.traceBatch();
                    renderer.updateResult();
                    window.setVisible(true);
                    Window::pollMessages();
                    for (int i = 0; i < 3; i++)
                    {
                        renderer.displayResult(gl);
                        gl->swapBuffers();
                    }

                    // Warm up and measure.

                    for (int i = 0; i < warmupRepeats; i++)
                        renderer.traceBatch();
                    for (int i = 0; i < measureRepeats; i++)
                        totalLaunchTime += renderer.traceBatch();
                }

                // Error => skip.

                if (hasError())
                    return;
            }

            // Calculate Mrays/s.

            F32 mraysPerSec = (F32)totalRays / totalLaunchTime * 1.0e-6f;
            results.add(mraysPerSec);
            printf("Mrays/s = %.2f\n", mraysPerSec);
            printf("\n");
        }
    }

    // Print summary table.

    printf("Done.\n");
    printf("\n");

    printf("%-42s", "Kernel");
    for (int i = 0; i < numRayTypes; i++)
        printf("%-10s", s_rayTypeNames[i]);
    printf("\n");

    printf("%-42s", "---");
    for (int i = 0; i < numRayTypes; i++)
        printf("%-10s", "---");
    printf("\n");

    for (int i = 0; i < kernels.getSize(); i++)
    {
        printf("%-42s", kernels[i].getPtr());
        for (int j = 0; j < numRayTypes; j++)
            printf("%-10.2f", results[i * numRayTypes + j]);
        printf("\n");
    }

    printf("%-42s", "---");
    for (int i = 0; i < numRayTypes; i++)
        printf("%-10s", "---");
    printf("\n");
    printf("\n");
}

//------------------------------------------------------------------------

void FW::init(void)
{
    // Parse mode.

    bool modeInteractive    = false;
    bool modeBenchmark      = false;
    bool showHelp           = false;

    if (argc < 2)
    {
        printf("Specify \"--help\" for a list of command-line options.\n\n");
        modeInteractive = true;
    }
    else
    {
        String mode = argv[1];
        if (mode == "interactive")      modeInteractive = true;
        else if (mode == "benchmark")   modeBenchmark = true;
        else                            showHelp = true;
    }

    // Parse options.

    String          logFile;
    Vec2i           frameSize       = Vec2i(1024, 768);
    String          stateFile;
    String          meshFile;
    Array<String>   cameras;
    Array<String>   kernels;
    F32             sbvhAlpha       = 1.0e-5f;
    F32             aoRadius        = 5.0f;
    int             numSamples      = 32;
    bool            sortRays        = true;
    int             warmupRepeats   = 2;
    int             measureRepeats  = 10;

    for (int i = 2; i < argc; i++)
    {
        const char* ptr = argv[i];

        if ((parseLiteral(ptr, "--help") || parseLiteral(ptr, "-h")) && !*ptr)
        {
            showHelp = true;
        }
        else if (parseLiteral(ptr, "--log="))
        {
            if (!*ptr)
                setError("Invalid log file '%s'!", argv[i]);
            logFile = ptr;
        }
        else if (parseLiteral(ptr, "--size="))
        {
            if (!parseInt(ptr, frameSize.x) || !parseLiteral(ptr, "x") || !parseInt(ptr, frameSize.y) || *ptr || min(frameSize) <= 0)
                setError("Invalid frame size '%s'!", argv[i]);
        }
        else if (modeInteractive && parseLiteral(ptr, "--state="))
        {
            if (!*ptr)
                setError("Invalid state file '%s'!", argv[i]);
            stateFile = ptr;
        }
        else if (modeBenchmark && parseLiteral(ptr, "--mesh="))
        {
            if (!*ptr)
                setError("Invalid mesh file '%s'!", argv[i]);
            meshFile = ptr;
        }
        else if (modeBenchmark && parseLiteral(ptr, "--camera="))
        {
            if (!*ptr)
                setError("Invalid camera signature '%s'!", argv[i]);
            cameras.add(ptr);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--kernel="))
        {
            if (!*ptr)
                setError("Invalid kernel name '%s'!", argv[i]);
            kernels.add(ptr);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--sbvh-alpha="))
        {
            if (!parseFloat(ptr, sbvhAlpha) || *ptr || sbvhAlpha < 0.0f || sbvhAlpha > 1.0f)
                setError("Invalid SBVH alpha '%s'!", argv[i]);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--ao-radius="))
        {
            if (!parseFloat(ptr, aoRadius) || *ptr || aoRadius < 0.0f)
                setError("Invalid AO radius '%s'!", argv[i]);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--samples="))
        {
            if (!parseInt(ptr, numSamples) || *ptr || numSamples < 1)
                setError("Invalid number of samples '%s'!", argv[i]);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--sort="))
        {
            int value = 0;
            if (!parseInt(ptr, value) || *ptr || value < 0 || value > 1)
                setError("Invalid ray sorting enable/disable '%s'!", argv[i]);
            sortRays = (value != 0);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--warmup-repeats="))
        {
            if (!parseInt(ptr, warmupRepeats) || *ptr || warmupRepeats < 0)
                setError("Invalid number of warmup repeats '%s'!", argv[i]);
        }
        else if (modeBenchmark && parseLiteral(ptr, "--measure-repeats="))
        {
            if (!parseInt(ptr, measureRepeats) || *ptr || measureRepeats < 1)
                setError("Invalid number of measurement repeats '%s'!", argv[i]);
        }
        else
        {
            setError("Invalid option '%s'!", argv[i]);
        }
    }

    // Show help.

    if (showHelp)
    {
        printf("%s", s_commandHelpText);
        exitCode = 1;
        clearError();
        return;
    }

    // Log file specified => start logging.

    if (logFile.getLength())
        pushLogFile(logFile);

    // Validate options.

    if (modeBenchmark)
    {
        if (!meshFile.getLength())
            setError("Mesh file (--mesh) not specified!");
        if (!cameras.getSize())
            setError("No camera signatures (--camera) specified!");
        if (!kernels.getSize())
            listKernels(kernels);
    }

    // Run.

    if (modeInteractive)
        runInteractive(frameSize, stateFile);

    if (modeBenchmark)
        runBenchmark(frameSize, meshFile, cameras, kernels, sbvhAlpha, aoRadius, numSamples, sortRays, warmupRepeats, measureRepeats);

    // Handle errors.

    if (hasError())
    {
        printf("Error: %s\n", getError().getPtr());
        exitCode = 1;
        clearError();
        return;
    }
}

//------------------------------------------------------------------------
