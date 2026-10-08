"""Run the production coverage kernel body on a synthetic one-pixel CUDA tile.

Requires Windows CUDA/NVRTC 12.8, no editor window or third-party Python packages.
Fixture structs replace raster packing/decoding only; this checks the actual
coverage .cuh staging, synchronization, depth, identities and mass reduction.
It does not validate the real renderer's StageData ABI/decoder.
"""
import ctypes as c
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA = Path(os.environ["CUDA_PATH"])
DLL_DIR = os.add_dll_directory(str(CUDA / "bin"))
BUILTINS = c.CDLL(str(CUDA / "bin" / "nvrtc-builtins64_128.dll"))
nv = c.CDLL(str(CUDA / "bin" / "nvrtc64_120_0.dll"))
cu = c.WinDLL("nvcuda.dll")
U = c.c_uint32
P = c.c_uint64


def check(result, label):
    if result:
        raise RuntimeError(f"{label}: error {result}")


class Args(c.Structure):
    _fields_ = [("showRing", c.c_bool)]


class Target(c.Structure):
    _fields_ = [("framebuffer", P), ("width", c.c_int), ("height", c.c_int)]


class Stage(c.Structure):
    _fields_ = [("x", c.c_float), ("y", c.c_float), ("depth", c.c_float), ("flags", U), ("color", U)]


SOURCE = r'''
#include <cooperative_groups.h>
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
namespace cg = cooperative_groups;
struct vec2 {
    float x,y;
    __device__ vec2(float a=0,float b=0):x(a),y(b){}
    __device__ vec2 operator/(float s)const{return {x/s,y/s};}
    __device__ vec2 operator-(vec2 b)const{return {x-b.x,y-b.y};}
};
__device__ float dot(vec2 a,vec2 b){return a.x*b.x+a.y*b.y;}
struct vec4 {float x,y,z,a; __device__ vec4(float b,float c,float d,float e):x(b),y(c),z(d),a(e){} __device__ vec4(){} };
struct Uniforms{bool showRing;};
struct CommonLaunchArgs{Uniforms uniforms;};
struct RenderTarget{uint64_t* framebuffer;int width,height;};
struct Tile{uint32_t firstIndex,lastIndex;};
struct StageData{vec2 imgPos_encoded;float depth;uint32_t flags,color;};
struct RasterizationData{vec2 dir1,dir2,imgpos;float depth;uint32_t flags;vec4 colors;};
__device__ void decode_stagedata(StageData s,vec2& a,vec2& b,float& d){a={1,0};b={0,1};d=s.depth;}
#define TILE_SIZE_3DGS 16
#define PREFETCH_COUNT 2
#define FLAGS_DISABLE_DEPTHWRITE 1
#include "render/gaussian_selector_coverage.cuh"
'''

nv.nvrtcCreateProgram.argtypes = [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p, c.c_int, c.c_void_p, c.c_void_p]
nv.nvrtcCompileProgram.argtypes = [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)]
for name in ("nvrtcGetProgramLogSize", "nvrtcGetPTXSize"):
    getattr(nv, name).argtypes = [c.c_void_p, c.POINTER(c.c_size_t)]
for name in ("nvrtcGetProgramLog", "nvrtcGetPTX"):
    getattr(nv, name).argtypes = [c.c_void_p, c.c_void_p]
cu.cuCtxCreate_v2.argtypes = [c.POINTER(c.c_void_p), U, c.c_int]
cu.cuModuleLoadData.argtypes = [c.POINTER(c.c_void_p), c.c_void_p]
cu.cuModuleGetFunction.argtypes = [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p]
cu.cuMemAlloc_v2.argtypes = [c.POINTER(P), c.c_size_t]
cu.cuMemFree_v2.argtypes = [P]
cu.cuMemcpyHtoD_v2.argtypes = [P, c.c_void_p, c.c_size_t]
cu.cuMemcpyDtoH_v2.argtypes = [c.c_void_p, P, c.c_size_t]
cu.cuLaunchKernel.argtypes = [c.c_void_p, U, U, U, U, U, U, U, c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p]
cu.cuModuleUnload.argtypes = [c.c_void_p]
cu.cuCtxDestroy_v2.argtypes = [c.c_void_p]
check(cu.cuInit(0), "cuInit")
device = c.c_int()
check(cu.cuDeviceGet(c.byref(device), 0), "device")
major, minor = c.c_int(), c.c_int()
check(cu.cuDeviceGetAttribute(c.byref(major), 75, device), "compute major")
check(cu.cuDeviceGetAttribute(c.byref(minor), 76, device), "compute minor")
context, module, kernel, program = (c.c_void_p() for _ in range(4))
check(cu.cuCtxCreate_v2(c.byref(context), 0, device), "context")
try:
    check(nv.nvrtcCreateProgram(c.byref(program), SOURCE.encode(), b"selector_fixture.cu", 0, None, None), "create NVRTC")
    options = (c.c_char_p * 4)(b"--std=c++17", f"--gpu-architecture=compute_{major.value}{minor.value}".encode(), f"-I{ROOT / 'src'}".encode(), f"-I{CUDA / 'include'}".encode())
    result = nv.nvrtcCompileProgram(program, len(options), options)
    size = c.c_size_t()
    nv.nvrtcGetProgramLogSize(program, c.byref(size))
    log = c.create_string_buffer(size.value)
    nv.nvrtcGetProgramLog(program, log)
    if result:
        raise RuntimeError(log.value.decode(errors="replace"))
    check(nv.nvrtcGetPTXSize(program, c.byref(size)), "PTX size")
    ptx = c.create_string_buffer(size.value)
    check(nv.nvrtcGetPTX(program, ptx), "PTX")
    check(cu.cuModuleLoadData(c.byref(module), ptx), "load module")
    check(cu.cuModuleGetFunction(c.byref(kernel), module, b"kernel_gaussian_selector_coverage"), "coverage kernel")

    def run(stop, alpha=(204, 128, 128, 128), rows=(0, 1, 2, 3), depths=(1, 2, 3, 4), opaque=100, masks=(1, 0), flags=0, tile_key=0, owner_budget=16):
        allocations = []

        def upload(value):
            pointer = P()
            check(cu.cuMemAlloc_v2(c.byref(pointer), c.sizeof(value)), "allocate")
            allocations.append(pointer)
            check(cu.cuMemcpyHtoD_v2(pointer, c.byref(value), c.sizeof(value)), "upload")
            return pointer

        try:
            tiles = upload((U * 2)(0, 3))
            tile_ids = upload((U * 4)(*[tile_key] * 4))
            indices = upload((U * 4)(0, 1, 2, 3))
            stages = upload((Stage * 4)(*[Stage(0, 0, d, flags, a << 24) for d, a in zip(depths, alpha)]))
            identities = upload((U * 4)(*rows))
            fg, bg = (upload((c.c_uint8 * 1)(m)) for m in masks)
            depth = upload((c.c_float * 1)(opaque))
            masses = [upload((c.c_float * 4)()) for _ in range(7)]
            args = [Args(False), Target(0, 1, 1), tiles, tile_ids, indices, stages, U(4), identities, fg, bg, depth, *masses[:3], c.c_float(stop), U(owner_budget), *masses[3:], c.c_float(.02)]
            parameters = (c.c_void_p * len(args))(*[c.addressof(a) for a in args])
            check(cu.cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, None, parameters, None), "launch")
            check(cu.cuCtxSynchronize(), "synchronize")
            output = []
            for mass in masses:
                value = (c.c_float * 4)()
                check(cu.cuMemcpyDtoH_v2(value, mass, c.sizeof(value)), "readback")
                output.append(list(value))
            return output
        finally:
            for pointer in allocations:
                cu.cuMemFree_v2(pointer)

    def near(a, b):
        assert abs(a-b) < 1e-6, (a, b)

    full = run(0)
    transmittance = 1.0
    for i, alpha in enumerate((204, 128, 128, 128)):
        mass = transmittance * alpha / 255
        near(full[0][i], mass)
        near(full[1][i], mass)
        transmittance -= mass
    for stop in (.5, .9):
        shallow = run(stop)
        near(shallow[0][0], .8)
        assert shallow[0][1:] == [0, 0, 0], "cutoff persists across batches"
        unpainted = run(stop, masks=(0, 0))
        assert unpainted[0] == shallow[0] and unpainted[1] == [0]*4, "whole-view denominator"
        negative = run(stop, masks=(0, 1))
        assert negative[2] == shallow[1], "FG/BG symmetry"
    terminated = run(0, alpha=(254, 128, 128, 128))
    assert terminated[0][1:] == [0]*3, "standard stop persists across batches"
    blocked = run(0, depths=(1, 3, 4, 5), opaque=2)
    assert blocked[0][1:] == [0]*3, "opaque depth persists across batches"
    outside = run(.5, rows=(0xffffffff, 0, 1, 2))
    assert outside[0] == [0]*4, "outside-domain occluder"
    no_depth = run(.5, flags=1)
    assert no_depth[0][1:] == [0]*3, "evidence cutoff independent of depthwrite"
    empty = run(0, tile_key=1)
    assert empty[0] == [0]*4, "empty tile"
    limited = run(0, owner_budget=2)
    assert limited[0][1]>0 and limited[0][2:]==[0, 0], "per-pixel owner budget"
    assert full[3]==[1]*4 and full[5]==[1, 0, 0, 0], "mask support and front center visibility"
    tolerant = run(0, depths=(1, 1.015, 1.03, 2))
    assert tolerant[5]==[1, 1, 0, 0], "2 percent relative center depth tolerance"
    print("CUDA coverage fixtures passed: 2 batches, standard/T=0.5/T=0.9, full-view denominator, FG/BG, opaque depth, outside occluder, no-depthwrite, empty tile")

    # Also compile the real renderer with its production types and NVRTC flags.
    # This is compile validation only (the synthetic fixtures above run on GPU).
    renderer_program = c.c_void_p()
    renderer_source = (ROOT / "src" / "gaussians_rendering.cu").read_bytes()
    check(nv.nvrtcCreateProgram(c.byref(renderer_program), renderer_source, b"gaussians_rendering.cu", 0, None, None), "create renderer NVRTC")
    try:
        values = ["--std=c++20", f"--gpu-architecture=compute_{major.value}{minor.value}",
                  "--use_fast_math", "--extra-device-vectorization", "--relocatable-device-code=true",
                  "-default-device", "--dlink-time-opt", "--disable-warnings",
                  "-DSPLATSHOP_ENABLE_GAUSSIAN_SELECTOR=1", f"-I{ROOT / 'src'}", f"-I{ROOT}",
                  f"-I{ROOT / 'include'}", f"-I{CUDA / 'include'}", f"-I{CUDA / 'include' / 'cuda' / 'std'}"]
        renderer_options = (c.c_char_p * len(values))(*[v.encode() for v in values])
        result = nv.nvrtcCompileProgram(renderer_program, len(values), renderer_options)
        if result:
            nv.nvrtcGetProgramLogSize(renderer_program, c.byref(size))
            renderer_log = c.create_string_buffer(size.value)
            nv.nvrtcGetProgramLog(renderer_program, renderer_log)
            raise RuntimeError(renderer_log.value.decode(errors="replace"))
        print("Production gaussians_rendering.cu NVRTC compilation passed (selector enabled)")
    finally:
        nv.nvrtcDestroyProgram(c.byref(renderer_program))
finally:
    if program:
        nv.nvrtcDestroyProgram(c.byref(program))
    if module:
        cu.cuModuleUnload(module)
    cu.cuCtxDestroy_v2(context)
