
// CHECKLICENSE
// much stuff from https://github.com/mkkellogg/GaussianSplats3D/blob/main/LICENSE
// (MIT license)

#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>

#define CUB_DISABLE_BF16_SUPPORT

#define USE_FAST_CONFIG

#define GLM_FORCE_CUDA
#define CUDA_VERSION 12000

namespace std{
	using size_t = ::size_t;
};

using namespace std;

#include <cuda_fp16.h>

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
// #include "./libs/glm/glm/gtc/matrix_access.hpp"
// #include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"
#include "./libs/glm/glm/gtx/matrix_decompose.hpp"

#include "utils.cuh"
#include "HostDeviceInterface.h"
#include "math.cuh"

namespace cg = cooperative_groups;

using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::quat;
using glm::dot;
using glm::transpose;
using glm::inverse;


constexpr int VIEWMODE_DESKTOP = 0;
constexpr int VIEWMODE_DESKTOP_VR = 1;
constexpr int VIEWMODE_IMMERSIVE_VR = 2;
// constexpr uint32_t BACKGROUND_COLOR = 0xff000000;
// constexpr uint32_t BACKGROUND_COLOR = 0xffffffff;
constexpr uint32_t BACKGROUND_COLOR = 0xff443322;
constexpr uint64_t DEFAULT_PIXEL = (uint64_t(Infinity) << 32) | BACKGROUND_COLOR;

constexpr float3 SPECTRAL[11] = {
	float3{158,1,66},
	float3{213,62,79},
	float3{244,109,67},
	float3{253,174,97},
	float3{254,224,139},
	float3{255,255,191},
	float3{230,245,152},
	float3{171,221,164},
	float3{102,194,165},
	float3{50,136,189},
	float3{94,79,162},
};

__device__ uint64_t g_uclosest;

__device__ double g_x;
__device__ double g_y;
__device__ double g_z;
__device__ uint32_t g_numSelected;
__device__ bool dbg_bool;
__device__ uint32_t g_counter;

constexpr int dbgtile_x = 60;
constexpr int dbgtile_y = 50;
constexpr float basisvector_encoding_factor = 20.0f;

glm::i16vec2 encode_basisvector_i16vec2(vec2 basisvector){

	float length = glm::length(basisvector);
	float angle = atan2(basisvector.y, basisvector.x);

	int16_t ilength = clamp(length * basisvector_encoding_factor, 0.0f, 60'000.0f);
	int16_t iangle = angle * 10'000.0f;

	return {iangle, ilength};
}

vec2 decode_basisvector_i16vec2(glm::i16vec2 encoded){

	float length = float(encoded.y) / basisvector_encoding_factor;
	float angle = float(encoded.x) / 10'000.0f;

	float x = cos(angle);
	float y = sin(angle);

	return vec2{x, y} * length;
}

__half2 decode_basisvector_i16vec2_half2(glm::i16vec2 encoded){

	float length = float(encoded.y) / basisvector_encoding_factor;
	float angle = float(encoded.x) / 10'000.0f;

	float x = cos(angle) * length;
	float y = sin(angle) * length;

	return __half2{x, y};
}

#if defined(STAGEDATA_16BYTE)

	void encode_stagedata(StageData& stagedata, vec2 a, vec2 b, float depth){
		stagedata.basisvector1_encoded = encode_basisvector_i16vec2(a);
		// stagedata.basisvector2_encoded = encode_basisvector_i16vec2(b);

		vec2 b_encoded = encode_basisvector_i16vec2(b);
		stagedata.basisvector2_encoded = b_encoded.y;
		
		// stagedata.depth = depth;

		__half hdepth = depth;
		memcpy(&stagedata.depth_encoded, &hdepth, 2);
	}

	void decode_stagedata(StageData stagedata, vec2& a, vec2& b, float& depth){
		a = decode_basisvector_i16vec2(stagedata.basisvector1_encoded);
		// b = decode_basisvector_i16vec2(stagedata.basisvector2_encoded);

		float b_length = float(stagedata.basisvector2_encoded) / basisvector_encoding_factor;
		b = normalize(vec2{-a.y, a.x}) * b_length;

		// depth = stagedata.depth;

		__half hdepth;
		memcpy(&hdepth, &stagedata.depth_encoded, 2);
		depth = hdepth;
	}

#elif defined(STAGEDATA_20BYTE)

	void encode_stagedata(StageData& stagedata, vec2 a, vec2 b, float depth){
		stagedata.basisvector1_encoded = encode_basisvector_i16vec2(a);
		stagedata.basisvector2_encoded = encode_basisvector_i16vec2(b);

		stagedata.depth = depth;
	}

	void decode_stagedata(StageData stagedata, vec2& a, vec2& b, float& depth){
		a = decode_basisvector_i16vec2(stagedata.basisvector1_encoded);
		b = decode_basisvector_i16vec2(stagedata.basisvector2_encoded);

		depth = stagedata.depth;
	}

#elif defined(STAGEDATA_24BYTE)

	void encode_stagedata(StageData& stagedata, vec2 a, vec2 b, float depth){
		stagedata.basisvector1_encoded = encode_basisvector_i16vec2(a);
		stagedata.basisvector2_encoded = encode_basisvector_i16vec2(b);

		stagedata.depth = depth;
	}

	void decode_stagedata(StageData stagedata, vec2& a, vec2& b, float& depth){
		a = decode_basisvector_i16vec2(stagedata.basisvector1_encoded);
		b = decode_basisvector_i16vec2(stagedata.basisvector2_encoded);

		depth = stagedata.depth;
	}

#endif

// based on https://github.com/sjtuzq/point-radiance/blob/main/modules/sh.py (Differentiable Point-Based Radiance Fields for Efficient View Synthesis)
// and Inria 3DGS forward.cu - computeColorFromSH() (3D Gaussian Splatting for Real-Time Radiance Field Rendering)
vec3 getHarmonics(
	int degree, 
	int numCoefficients, 
	vec3 camdir,
	vec3* sh
){
	// SH coefficients from https://github.com/sjtuzq/point-radiance/blob/main/modules/sh.py
	// (BSD 2-clause license), Copyright 2021 The PlenOctree Authors.
	// constexpr float C0 = 0.28209479177387814;
	constexpr float C1 = 0.4886025119029199;
	constexpr float C2[] = {
		 1.0925484305920792f,
		-1.0925484305920792f,
		 0.31539156525252005f,
		-1.0925484305920792f,
		 0.5462742152960396f
	};
	constexpr float C3[] = {
		-0.5900435899266435f,
		 2.890611442640554f,
		-0.4570457994644658f,
		 0.3731763325901154f,
		-0.4570457994644658f,
		 1.445305721320277f,
		-0.5900435899266435f
	};
	constexpr float C4[] = {
		 2.5033429417967046f,
		-1.7701307697799304f,
		 0.9461746957575601f,
		-0.6690465435572892f,
		 0.10578554691520431f,
		-0.6690465435572892f,
		 0.47308734787878004f,
		-1.7701307697799304f,
		 0.6258357354491761f,
	};

	float xx = camdir.x * camdir.x;
	float yy = camdir.y * camdir.y;
	float zz = camdir.z * camdir.z;
	float xy = camdir.x * camdir.y; 
	float yz = camdir.y * camdir.z; 
	float xz = camdir.x * camdir.z;

	vec3 result = {0.0f, 0.0f, 0.0f};

	if(degree > 0){
		result = result -
			C1 * camdir.y * sh[0] + 
			C1 * camdir.z * sh[1] - 
			C1 * camdir.x * sh[2];
	}

	if(degree > 1){
		result = result +
			C2[0] * xy * sh[3] +
			C2[1] * yz * sh[4] +
			C2[2] * (2.0f * zz - xx - yy) * sh[5] +
			C2[3] * xz * sh[6] +
			C2[4] * (xx - yy) * sh[7];
	}

	if(degree > 2){
		result = result +
			C3[0] * camdir.y * (3.0f * xx - yy) * sh[8] +
			C3[1] * xy * camdir.z * sh[9] +
			C3[2] * camdir.y * (4.0f * zz - xx - yy) * sh[10] +
			C3[3] * camdir.z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
			C3[4] * camdir.x * (4.0f * zz - xx - yy) * sh[12] +
			C3[5] * camdir.z * (xx - yy) * sh[13] +
			C3[6] * camdir.x * (xx - 3.0f * yy) * sh[14];
	}

	return result;
}



// We need the same iteration logic for "touched" tiles in multiple kernels. 
template <typename Function>
void forEachTouchedTile(vec2 splatCoord, vec2 basisVector1, vec2 basisVector2, RenderTarget target, Function f){

	// forEachTouchedTile_approx(splatCoord, basisVector1, basisVector2, target, f);

	// use approximate tile-splat intersection
	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	ivec2 tile_start = {
		(splatCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(splatCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));

	if(tile_end.x < 0 || tile_start.x >= tiles_x) return;
	if(tile_end.y < 0 || tile_start.y >= tiles_y) return;

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	// float diag = 22.627416997969522f;  // sqrt(16.0f * 16.0f + 16.0f * 16.0f);
	float tileRadius = 11.313708498984761f + 0.0f; // sqrt(8.0f * 8.0f + 8.0f * 8.0f);
	for(int tile_x = tile_start.x; tile_x <= tile_end.x; tile_x++)
	for(int tile_y = tile_start.y; tile_y <= tile_end.y; tile_y++)
	{
		vec2 tilePos = {tile_x * 16.0f + 8.0f, tile_y * 16.0f + 8.0f}; 

		bool intersectsTile = intersection_circle_splat(
			tilePos, tileRadius, 
			splatCoord, 
			basisVector1, 
			basisVector2
		);

		// intersectsTile = true;

		if(intersectsTile){
			f(tile_x, tile_y);
		}
	}
}

template <typename Function>
void forEachTouchedTile(int mode, vec2 splatCoord, vec2 basisVector1, vec2 basisVector2, RenderTarget target, Function f){

	// if(cg::this_grid().thread_rank() == 0) printf("mode: %d \n", mode);

	if(mode == INTERSECTION_APPROXIMATE){
		forEachTouchedTile_approx(splatCoord, basisVector1, basisVector2, target, f);
	}else if(mode == INTERSECTION_3DGS){
		forEachTouchedTile_3dgs(splatCoord, basisVector1, basisVector2, target, f);
	}else if(mode == INTERSECTION_TIGHTBB){
		forEachTouchedTile_tightBB(splatCoord, basisVector1, basisVector2, target, f);
	}

}

// We need the same iteration logic for "touched" tiles in multiple kernels. 
template <typename Function>
void forEachTouchedTile_approx(vec2 splatCoord, vec2 basisVector1, vec2 basisVector2, RenderTarget target, Function f){

	// use approximate tile-splat intersection
	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	ivec2 tile_start = {
		(splatCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(splatCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));

	if(tile_end.x < 0 || tile_start.x >= tiles_x) return;
	if(tile_end.y < 0 || tile_start.y >= tiles_y) return;

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	// float diag = 22.627416997969522f;  // sqrt(16.0f * 16.0f + 16.0f * 16.0f);
	float tileRadius = 11.313708498984761f + 0.0f; // sqrt(8.0f * 8.0f + 8.0f * 8.0f);
	for(int tile_x = tile_start.x; tile_x <= tile_end.x; tile_x++)
	for(int tile_y = tile_start.y; tile_y <= tile_end.y; tile_y++)
	{
		vec2 tilePos = {tile_x * 16.0f + 8.0f, tile_y * 16.0f + 8.0f}; 

		bool intersectsTile = intersection_circle_splat(
			tilePos, tileRadius, 
			splatCoord, 
			basisVector1, 
			basisVector2
		);

		// intersectsTile = true;

		if(intersectsTile){
			f(tile_x, tile_y);
		}
	}
}

// We need the same iteration logic for "touched" tiles in multiple kernels. 
template <typename Function>
void forEachTouchedTile_3dgs(vec2 splatCoord, vec2 basisVector1, vec2 basisVector2, RenderTarget target, Function f){

	// Use 3DGS-style bounding box around ellipses' bounding sphere
	float maxRadius = max(length(basisVector1), length(basisVector2));

	ivec2 tile_start = {
		(splatCoord.x - maxRadius) / TILE_SIZE_3DGS,
		(splatCoord.y - maxRadius) / TILE_SIZE_3DGS,
	};
	ivec2 tile_end = {
		(splatCoord.x + maxRadius) / TILE_SIZE_3DGS,
		(splatCoord.y + maxRadius) / TILE_SIZE_3DGS,
	};

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));

	if(tile_end.x < 0 || tile_start.x >= tiles_x) return;
	if(tile_end.y < 0 || tile_start.y >= tiles_y) return;

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	for(int tile_x = tile_start.x; tile_x <= tile_end.x; tile_x++)
	for(int tile_y = tile_start.y; tile_y <= tile_end.y; tile_y++)
	{
		f(tile_x, tile_y);
	}
}

template <typename Function>
void forEachTouchedTile_tightBB(vec2 splatCoord, vec2 basisVector1, vec2 basisVector2, RenderTarget target, Function f){

	// Use tight bounding box around ellipse
	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	ivec2 tile_start = {
		(splatCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(splatCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(splatCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));

	if(tile_end.x < 0 || tile_start.x >= tiles_x) return;
	if(tile_end.y < 0 || tile_start.y >= tiles_y) return;

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	for(int tile_x = tile_start.x; tile_x <= tile_end.x; tile_x++)
	for(int tile_y = tile_start.y; tile_y <= tile_end.y; tile_y++)
	{
		f(tile_x, tile_y);
	}
}



bool isDebugTile(int x, int y){
	return x == dbgtile_x && y == dbgtile_y;
}

// Shared multi-view preprocessing. The object-space covariance is independent
// of the camera, so build it once and let all four projection stages reuse it.
extern "C" __global__
void kernel_buildCov3D(GaussianData model){
	int splatIndex = cg::this_grid().thread_rank();
	if(splatIndex >= model.count || model.cov3d == nullptr) return;

	vec4 quat = model.quaternion[splatIndex];
	mat3 rotation = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));
	mat3 scale = mat3(1.0f);
	scale[0][0] = model.scale[splatIndex].x;
	scale[1][1] = model.scale[splatIndex].y;
	scale[2][2] = model.scale[splatIndex].z;
	mat3 covariance = rotation * scale * scale * transpose(rotation);

	Cov3DElements packed;
	packed.m11 = covariance[0][0];
	packed.m12 = covariance[0][1];
	packed.m13 = covariance[0][2];
	packed.m22 = covariance[1][1];
	packed.m23 = covariance[1][2];
	packed.m33 = covariance[2][2];
	model.cov3d[splatIndex] = packed;
}

// stages the model for rasterization, meaning it creates tile fragments for each tile the splat overlaps, and adds it to the staging buffer.
// Much of the math here is from https://github.com/mkkellogg/GaussianSplats3D
extern "C" __global__
void kernel_stageSplats(
	CommonLaunchArgs args,
	RenderTarget target,
	ColorCorrection colorCorrection,
	GaussianData model,
	// out
	uint32_t* visibleSplatCounter,
	uint32_t* numTilefragments,
	uint32_t* numTilefragments_splatwise,
	float* staging_depth,
	StageData* staging_data,
	uint32_t* ordering
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	int splatIndex = grid.thread_rank();

	if(splatIndex >= model.count) return;

	mat4 world = model.transform;
	mat4 view = target.view;
	mat4 proj = target.proj;
	mat4 worldView = view * world;
	mat4 worldViewProj = proj * view * world;

	vec3 splatPos = model.position[splatIndex];
	vec4 worldPos = world * vec4(splatPos, 1.0f);
	vec4 viewPos = view * worldPos;
	vec4 ndc = proj * viewPos;

	ndc.x = ndc.x / ndc.w;
	ndc.y = ndc.y / ndc.w;
	ndc.z = ndc.z / ndc.w;


	// if(grid.thread_rank() == 0) printf("=============\n");
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.position));
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.scale));
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.quaternion));
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.color));
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.sphericalHarmonics));
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.cov3d));
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.depth));
	// if(grid.thread_rank() == 0) printf("%llu\n", uint64_t(model.flags));

	float cameraDepth = -viewPos.z;
	if(target.projectionMode == PROJECTION_PERSPECTIVE && ndc.w <= 0.0f) return;
	if(target.projectionMode == PROJECTION_ORTHOGRAPHIC && cameraDepth <= 0.0f) return;
	// Early-Discard gaussians whose position is far outside the frustum 
	if(ndc.x < -1.5f || ndc.x >  1.5f) return;
	if(ndc.y < -1.5f || ndc.y >  1.5f) return;

	vec4 color = model.color[splatIndex].normalized();

	// WIP: SHs not yet robust under model and splat transformations
	constexpr int updateOverXFrames = 10;
	// if(false)
	if(splatIndex % updateOverXFrames == args.uniforms.frameCount % updateOverXFrames)
	if(!target.isRight)
	if(model.shDegree > 0 ){
		// if(splatIndex == 0) printf("model.numSHCoefficients: %d \n", model.numSHCoefficients);
		// int64_t offset = splatIndex * 45;
		int64_t offset = splatIndex * model.numSHCoefficients;
		vec3* shs = (vec3*)(model.sphericalHarmonics + offset);
		vec3 camPos = inverse(target.view) * vec4(0.0f, 0.0f, 0.0f, 1.0f);
		

		vec4 harmonics = vec4{0.0f};
		// camdir = normalize(vec3{
		// 	cos(0.5f * floor(args.uniforms.time)), 
		// 	sin(0.5f * floor(args.uniforms.time)), 
		// 	 0.0f
		// });

		vec3 camdir = normalize(vec3(worldPos) - camPos);
		camdir = vec3((inverse(model.transform)) * vec4(camdir, 0.0f));
		camdir = normalize(camdir);
		harmonics = vec4(getHarmonics(model.shDegree, model.numSHCoefficients, camdir, shs), 0.0f);

		// vec3 d_scale;
		// quat d_rotation;
		// vec3 d_translation;
		// vec3 d_skew;
		// vec4 d_perspective;
		// glm::decompose(model.transform, d_scale, d_rotation, d_translation, d_skew, d_perspective);
		// d_rotation = glm::conjugate(d_rotation);

		// mat3 rotation = glm::mat3_cast(d_rotation);

		// vec3 shs_transformed[15];
		// rotateSH(model.shDegree, shs, shs_transformed, rotation);

		// harmonics = vec4(getHarmonics(model.shDegree, model.numSHCoefficients, camdir, shs_transformed), 0.0f);

		vec4 colBaked = color + 1.0f * harmonics;
		// vec4 colBaked = 50.0f * harmonics;
		colBaked = glm::max(colBaked, 0.0f);
		colBaked.a = color.a;

		uint32_t C = 
			(uint32_t(clamp(256.0f * colBaked.r, 0.0f, 255.0f)) <<  0) |
			(uint32_t(clamp(256.0f * colBaked.g, 0.0f, 255.0f)) <<  8) |
			(uint32_t(clamp(256.0f * colBaked.b, 0.0f, 255.0f)) << 16) |
			(uint32_t(clamp(256.0f * colBaked.a, 0.0f, 255.0f)) << 24);
		model.color_resolved[splatIndex] = C;

		// color = colBaked;
	}

	if(model.shDegree > 0 )
	{
		uint32_t C = model.color_resolved[splatIndex];

		color.r = float((C >>  0) & 0xff) / 256.0f;
		color.g = float((C >>  8) & 0xff) / 256.0f;
		color.b = float((C >> 16) & 0xff) / 256.0f;
		color.a = float((C >> 24) & 0xff) / 256.0f;
	}


	vec4 quat = model.quaternion[splatIndex];

	// int8_t qw = quat.w * 127.0f;

	// // __half qx = __half(quat.x);
	// // __half qy = __half(quat.y);
	// // __half qz = __half(quat.z);
	// // __half qw = __half(quat.w);

	auto q = glm::quat(quat.x, quat.y, quat.z, quat.w);
	mat3 rotation = glm::mat3_cast(q);

	mat3 scale = mat3(1.0f);
	scale[0][0] = model.scale[splatIndex].x;
	scale[1][1] = model.scale[splatIndex].y;
	scale[2][2] = model.scale[splatIndex].z;

	mat3 cov3D;
	if(target.usePrecomputedCov3D && model.cov3d != nullptr){
		Cov3DElements packed = model.cov3d[splatIndex];
		cov3D = mat3(
			packed.m11, packed.m12, packed.m13,
			packed.m12, packed.m22, packed.m23,
			packed.m13, packed.m23, packed.m33);
	}else{
		cov3D = rotation * scale * scale * transpose(rotation);
	}

	// RaDe-GS: build the ray-space plane once per Gaussian.  The standard
	// renderer below evaluates this plane for every Gaussian-pixel pair.
	// OpenGL view space looks along -Z, so flip that axis before applying the
	// ray-space derivation (whose positive third axis is camera distance).
	vec3 rayPlane = vec3(0.0f);
	vec3 normal = vec3(0.0f, 0.0f, -1.0f);
	if(args.uniforms.enableRaDeGeometry){
		vec2 rayFocal = vec2(
			proj[0][0] * target.width * 0.5f,
			proj[1][1] * target.height * 0.5f);
		vec3 rayCenter = vec3(viewPos.x, viewPos.y, -viewPos.z);
		float zc = rayCenter.z;
		float tc = length(rayCenter);
		float minimumScale = min(model.scale[splatIndex].x, min(model.scale[splatIndex].y, model.scale[splatIndex].z));
		if(zc > 1e-6f && tc > 1e-6f && minimumScale > 1e-6f){
			mat3 cameraFlip(1.0f);
			cameraFlip[2][2] = -1.0f;
			mat3 modelToRayCamera = cameraFlip * mat3(worldView);
			// Construct Sigma'_inv as A^T A instead of directly inverting the
			// covariance. This mirrors RaDe-GS and is stable for thin splats.
			mat3 inverseScale(1.0f);
			inverseScale[0][0] = 1.0f / model.scale[splatIndex].x;
			inverseScale[1][1] = 1.0f / model.scale[splatIndex].y;
			inverseScale[2][2] = 1.0f / model.scale[splatIndex].z;
			mat3 inverseCovarianceFactor = inverseScale * transpose(rotation) * inverse(modelToRayCamera);
			mat3 inverseCovarianceRayCamera = transpose(inverseCovarianceFactor) * inverseCovarianceFactor;

			float u = rayCenter.x / zc;
			float v = rayCenter.y / zc;
			vec3 uvh = vec3(u, v, 1.0f);
			vec3 q = inverseCovarianceRayCamera * uvh;
			float qLength = length(q);
			if(qLength > 1e-8f && isfinite(qLength)){
				vec3 qNormalized = q / qLength;
				float u2 = u * u;
				float v2 = v * v;
				float uv = u * v;
				float rayLength2 = u2 + v2 + 1.0f;
				float factor = tc / rayLength2;
				float denominator = max(dot(qNormalized, uvh), 1e-7f);
				mat3 nJInverse = mat3(
					v2 + 1.0f, -uv, 0.0f,
					-uv, u2 + 1.0f, 0.0f,
					-u, -v, 0.0f);
				vec3 plane = nJInverse * (qNormalized / denominator);
				rayPlane = vec3(plane.x * factor / rayFocal.x, plane.y * factor / rayFocal.y, tc);

				vec3 rayNormal = vec3(-plane.x * factor, -plane.y * factor, -1.0f);
				mat3 nJ = mat3(
					1.0f / zc, 0.0f, -rayCenter.x / (zc * zc),
					0.0f, 1.0f / zc, -rayCenter.y / (zc * zc),
					rayCenter.x / tc, rayCenter.y / tc, rayCenter.z / tc);
				vec3 surfaceNormal = nJ * rayNormal;
				float normalLength = length(surfaceNormal);
				if(normalLength > 1e-8f && isfinite(normalLength)) normal = surfaceNormal / normalLength;
			}
		}
	}

	// check lower bitrates
	// float* floats = (float*)&cov3D;
	// for(int i = 0; i < 9; i++){
	// 	floats[i] = __half(floats[i]);
	// 	// floats[i] = clamp(floats[i], -10.0f, 10.0f);
	// }

	// model.cov3d[index].m11 = cov3D[0][0];
	// model.cov3d[index].m12 = cov3D[0][1];
	// model.cov3d[index].m13 = cov3D[0][2];
	// model.cov3d[index].m22 = cov3D[1][1];
	// model.cov3d[index].m23 = cov3D[1][2];
	// model.cov3d[index].m33 = cov3D[2][2];

	// auto cov3Del = model.cov3d[index];
	// mat3 cov3D;
	// cov3D[0][0] = cov3Del.m11;
	// cov3D[0][1] = cov3Del.m12;
	// cov3D[0][2] = cov3Del.m13;

	// cov3D[1][0] = cov3Del.m12;
	// cov3D[1][1] = cov3Del.m22;
	// cov3D[1][2] = cov3Del.m23;

	// cov3D[2][0] = cov3Del.m13;
	// cov3D[2][1] = cov3Del.m23;
	// cov3D[2][2] = cov3Del.m33;

	vec2 focal = vec2(
		proj[0][0] * target.width * 0.5f,
		proj[1][1] * target.height * 0.5f
	);
	mat3 J;
	if(target.projectionMode == PROJECTION_ORTHOGRAPHIC){
		// Constant orthographic Jacobian. With axis-aligned front/top/side
		// cameras this reduces to selecting the corresponding Cov3D components.
		J = mat3(
			focal.x, 0.0f, 0.0f,
			0.0f, focal.y, 0.0f,
			0.0f, 0.0f, 0.0f);
	}else{
		float s = 1.0f / (viewPos.z * viewPos.z);
		J = mat3(
			focal.x / viewPos.z, 0.0f, -(focal.x * viewPos.x) * s,
			0.0f, focal.y / viewPos.z, -(focal.y * viewPos.y) * s,
			0.0f, 0.0f, 0.0f);
	}

	mat3 W = transpose(mat3(worldView));
	mat3 T = W * J;

	mat3 cov2Dm = transpose(T) * cov3D * T;
	cov2Dm[0][0] += 0.3f;
	cov2Dm[1][1] += 0.3f;

	vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

	float a = cov2Dv.x;
	float d = cov2Dv.z;
	float b = cov2Dv.y;
	float D = a * d - b * b;
	float trace = a + d;
	float traceOver2 = 0.5f * trace;
	float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
	float eigenValue1 = traceOver2 + term2;		// λ₁
	float eigenValue2 = traceOver2 - term2;		// λ₂

	// eigenValue1 = max(eigenValue1, 0.1f);
	// eigenValue2 = max(eigenValue2, 0.1f);

	if(args.uniforms.cullSmallSplats){
		// clip tiny gaussians
		if(eigenValue1 < 0.02f) return;
		if(eigenValue2 < 0.02f) return;
	}else{
		eigenValue1 = max(eigenValue1, 0.001f);
		eigenValue2 = max(eigenValue2, 0.001f);

		// reduce opacity of small splats
		if(eigenValue1 == 0.001f) color.a *= 0.15f;
		if(eigenValue2 == 0.001f) color.a *= 0.15f;

		// cull small splats with little opacity?
	}
	// if(color.a < 10.0f / 256.0f) return;

	if(args.uniforms.makePoints){
		if(eigenValue1 < 0.105f) return;
		if(eigenValue2 < 0.105f) return;
	}

	vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
	vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

	float splatScale = args.uniforms.splatSize;
	const float sqrt8 = sqrt(8.0f);

	vec2 _basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), MAX_SCREENSPACE_SPLATSIZE);
	vec2 _basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), MAX_SCREENSPACE_SPLATSIZE);

	// We are using quantized basisvectors elsewhere, 
	// so we must make sure to also use them in this kernel to obtain matching results
	vec2 basisVector1, basisVector2;
	StageData tmp;
	float tmpdepth;
	encode_stagedata(tmp, _basisVector1, _basisVector2, tmpdepth);
	decode_stagedata(tmp, basisVector1, basisVector2, tmpdepth);

	// discard small splats with low opacity
	if(args.uniforms.cullSmallSplats)
	if(length(_basisVector1) < 5.0f)
	if(length(_basisVector2) < 5.0f)
	{
		if(color.a < 0.2f) return;
	}

	float depth = target.projectionMode == PROJECTION_ORTHOGRAPHIC ? cameraDepth : ndc.w;
	vec2 _pixelCoord = {
		((ndc.x) * 0.5f + 0.5f) * target.width,
		((ndc.y) * 0.5f + 0.5f) * target.height
	};

	glm::i16vec2 pixelCoord_encoded = _pixelCoord * 10.0f;
	vec2 pixelCoord = vec2(pixelCoord_encoded) / 10.0f;

	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	ivec2 tile_start = {
		(pixelCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(pixelCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_size = (tile_end - tile_start) + 1;

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));
	vec2 tileCoord = {
		pixelCoord.x / TILE_SIZE_3DGS,
		pixelCoord.y / TILE_SIZE_3DGS
	};

	if(tile_end.x < 0 || tile_start.x >= tiles_x) return;
	if(tile_end.y < 0 || tile_start.y >= tiles_y) return;

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	color = applyColorCorrection(color, colorCorrection);

	uint32_t flags = model.flags[splatIndex];

	{ // HANDLE FLAGS

		bool isSelected           = (flags & FLAGS_SELECTED) != 0;
		bool isDeleted            = (flags & FLAGS_DELETED) != 0;
		bool isHighlighted        = (flags & FLAGS_HIGHLIGHTED) != 0;
		bool isHighlightedNeg     = (flags & FLAGS_HIGHLIGHTED_NEGATIVE) != 0;

		auto rect = args.rectselect;
		if(isDeleted && isHighlighted && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			color.b = min(color.b * 2.0f + 0.4f, 1.0f);
			color.a = color.a / 4.0f;
		}else if(isDeleted){
			color.a = 0;
		}else if(isSelected && !isHighlightedNeg){
			color.r = min(color.r * 2.0f + 0.4f, 1.0f);
		}

		if(isHighlighted && (args.brush.mode == BRUSHMODE::ERASE)){
			color.r = min(color.r /  1.0f, 1.0f);
			color.g = min(color.g /  1.0f, 1.0f);
			color.b = min(color.b /  1.0f, 1.0f);
			color.a = min(color.a / 10.0f, 1.0f);
		}else if(isHighlighted && (args.brush.mode == BRUSHMODE::SELECT || rect.active)){
			color.r = min(color.r * 2.0f + 0.4f, 1.0f);
			color.g = min(color.g * 0.5f + 0.1f, 1.0f);
			color.b = min(color.b * 0.5f + 0.1f, 1.0f);
		}else if(isHighlighted && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			color.r = min(color.r * 0.5f + 0.1f, 1.0f);
			color.g = min(color.g * 0.5f + 0.1f, 1.0f);
			color.b = min(color.b * 2.0f + 0.4f, 1.0f);
		}
	}

	if(!model.writeDepth){
		flags = flags | FLAGS_DISABLE_DEPTHWRITE;
	}

	if((flags & FLAGS_DELETED) != 0) return;

	// TODO: check if its more efficient to run a separate kernel that 
	// initializes <numVisibleSplats> elements with ordering[index] = index;
	uint32_t visibleSplatID = atomicAdd(visibleSplatCounter, 1);
	ordering[visibleSplatID] = visibleSplatID;
	#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR
	if(target.selectorStageRows){
		target.selectorStageRows[visibleSplatID] = model.selectorTarget
			? (model.selectorLocalToSnapshot ? model.selectorLocalToSnapshot[uint32_t(splatIndex)] : model.selectorSnapshotBase + uint32_t(splatIndex))
			: 0xffffffffu;
	}
	if(model.selectorPreviewWords && model.selectorTarget){
		const uint64_t word = model.selectorPreviewWords[uint32_t(splatIndex) >> 6];
		if((word >> (uint32_t(splatIndex) & 63u)) & 1ull){
			// Preview is an independent visualization channel; it never mutates
			// FLAGS_SELECTED or the stored Gaussian color/SH.
			color = mix(color, vec4(0.05f, 1.0f, 0.20f, color.a), 0.65f);
		}
	}
	#endif

	StageData stuff;
	stuff.imgPos_encoded       = pixelCoord_encoded;
	stuff.flags                = flags;
	stuff.color                = to32BitColor(color);
	stuff.rayPlane             = rayPlane;
	stuff.normal               = normal;
	
	encode_stagedata(stuff, _basisVector1, _basisVector2, depth);

	staging_depth[visibleSplatID] = depth;
	staging_data[visibleSplatID] = stuff;

	// Count tile fragments that each splat produces
	uint32_t tileFrags = 0;

	
	// forEachTouchedTile(pixelCoord, basisVector1, basisVector2, target, [&](uint32_t tile_x, uint32_t tile_y){
	forEachTouchedTile(args.uniforms.intersectionMode, pixelCoord, basisVector1, basisVector2, target, [&](uint32_t tile_x, uint32_t tile_y){
		tileFrags++;
	});

	numTilefragments_splatwise[visibleSplatID] = tileFrags;
	atomicAdd(numTilefragments, tileFrags);
}

inline constexpr int K = 16;
// #define USE_FAST_CONFIG
#ifdef USE_FAST_CONFIG
// fast config
inline constexpr float MIN_ALPHA_THRESHOLD_RCP = 20.0f;
inline constexpr float MAX_CUTTOFF_SQ = 5.99146454711f; // logf(MIN_ALPHA_THRESHOLD_RCP * MIN_ALPHA_THRESHOLD_RCP)
#else
// reference config
inline constexpr float MIN_ALPHA_THRESHOLD_RCP = 255.0f;
inline constexpr float MAX_CUTTOFF_SQ = 11.0825270903f; // logf(MIN_ALPHA_THRESHOLD_RCP * MIN_ALPHA_THRESHOLD_RCP)
#endif
inline constexpr float MIN_ALPHA_THRESHOLD = 1.0f / MIN_ALPHA_THRESHOLD_RCP;
inline constexpr float MIN_ALPHA_THRESHOLD_CORE = (MIN_ALPHA_THRESHOLD_RCP >= 20.0f) ? 1.0f / 20.0f : MIN_ALPHA_THRESHOLD;
inline constexpr float MAX_FRAGMENT_ALPHA = 1.0f; // 3dgs uses 0.99f
inline constexpr float TRANSMITTANCE_THRESHOLD = 1e-4f;

extern "C" __global__
// void __launch_bounds__(256, 6) kernel_stageSplats_perspectivecorrect(
void kernel_stageSplats_perspectivecorrect(
	CommonLaunchArgs args,
	RenderTarget target,
	ColorCorrection colorCorrection,
	GaussianData model,
	// out
	uint32_t* visibleSplatCounter,
	uint32_t* numTilefragments,
	uint32_t* numTilefragments_splatwise,
	glm::i16vec4* staging_bounds,
	StageData_perspectivecorrect* staging_data,
	uint32_t* ordering
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	int splatIndex = grid.thread_rank();
	constexpr int BLOCK_SIZE = TILE_SIZE_PERSPCORRECT * TILE_SIZE_PERSPCORRECT;

	if(splatIndex >= model.count) return;

	mat4 world = model.transform;
	mat4 view = target.view;
	mat4 worldView = view * world;

	vec4 splatPos = vec4(model.position[splatIndex], 1.0f);
	vec4 M3 = -glm::transpose(worldView)[2]; // times -1 because camera looks in negative z direction

	float z_view = dot(M3, splatPos);
	if (z_view < 0.2f || z_view > 1000.0f) return;

	vec4 color = model.color[splatIndex].normalized();
	if (color.a < MIN_ALPHA_THRESHOLD) return;

	// WIP: SHs not yet robust under model and splat transformations
	constexpr int updateOverXFrames = 10;
	if (splatIndex % updateOverXFrames == args.uniforms.frameCount % updateOverXFrames)
	if (model.shDegree > 0) {
		int64_t offset = splatIndex * model.numSHCoefficients;
		vec3* shs = (vec3*)(model.sphericalHarmonics + offset);

		vec3 camPos_model = -worldView[3] * mat3(worldView); // same as inverse(worldView)[3] -> differences are smaller than 0.000001f
		vec3 viewdir = normalize(vec3(splatPos) - camPos_model);
		vec4 harmonics = vec4(getHarmonics(model.shDegree, model.numSHCoefficients, viewdir, shs), 0.0f);
		
		vec4 colBaked = color + harmonics;
		colBaked = glm::max(colBaked, 0.0f);
		colBaked.a = color.a;

		uint32_t C = 
			(uint32_t(clamp(256.0f * colBaked.r, 0.0f, 255.0f)) <<  0) |
			(uint32_t(clamp(256.0f * colBaked.g, 0.0f, 255.0f)) <<  8) |
			(uint32_t(clamp(256.0f * colBaked.b, 0.0f, 255.0f)) << 16) |
			(uint32_t(clamp(256.0f * colBaked.a, 0.0f, 255.0f)) << 24);
		model.color_resolved[splatIndex] = C;
	}

	if (model.shDegree > 0) {
		uint32_t C = model.color_resolved[splatIndex];
		color.r = float((C >>  0) & 0xff) / 256.0f;
		color.g = float((C >>  8) & 0xff) / 256.0f;
		color.b = float((C >> 16) & 0xff) / 256.0f;
		color.a = float((C >> 24) & 0xff) / 256.0f;
	}

	vec4 quat = model.quaternion[splatIndex];
	mat3 rotation = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));

	mat3 scale = mat3(0.0f);
	scale[0][0] = model.scale[splatIndex].x;
	scale[1][1] = model.scale[splatIndex].y;
	scale[2][2] = model.scale[splatIndex].z;

	mat4 VP = target.VP;
	mat4 VPM = VP * worldView;
	mat3 RS = rotation * scale;
	mat4 T = mat4(RS);
	T[3] = splatPos;
	mat4 VPMT = glm::transpose(VPM * T);

	// tight cutoff for the used opacity threshold
	float rho_cutoff = 2.0f * logf(color.a * MIN_ALPHA_THRESHOLD_RCP);
	vec4 d = vec4(rho_cutoff, rho_cutoff, rho_cutoff, -1.0f);
	vec4 VPMT4 = VPMT[3];
	float s = dot(d, VPMT4 * VPMT4);
	if (s == 0.0f) return;
	vec4 f = (1.0f / s) * d;
	// start with z-extent in screen-space for exact near/far plane culling
	vec4 VPMT3 = VPMT[2];
	float center_z = dot(f, VPMT3 * VPMT4);
	float extent_z = sqrtf(fmaxf(center_z * center_z - dot(f, VPMT3 * VPMT3), 0.0f));
	float z_min = center_z - extent_z;
	float z_max = center_z + extent_z;
	if (z_min < -1.0f || z_max > 1.0f) return;
	// now x/y-extent of the screen-space bounding box
	vec4 VPMT1 = VPMT[0];
	vec4 VPMT2 = VPMT[1];
	vec2 center = vec2(dot(f, VPMT1 * VPMT4), dot(f, VPMT2 * VPMT4));
	vec2 extent = vec2(
		sqrtf(fmaxf(center.x * center.x - dot(f, VPMT1 * VPMT1), 0.0f)),
		sqrtf(fmaxf(center.y * center.y - dot(f, VPMT2 * VPMT2), 0.0f))
	);

	// if(args.uniforms.cullSmallSplats){
	// 	if(extent.x < 0.5f || extent.y < 0.5f) return;
	// }

	// compute screen-space bounding box in tile coordinates (+0.5 to account for half-pixel shift in V)
	int tiles_x = (target.width + TILE_SIZE_PERSPCORRECT - 1) / TILE_SIZE_PERSPCORRECT;
	int tiles_y = (target.height + TILE_SIZE_PERSPCORRECT - 1) / TILE_SIZE_PERSPCORRECT;
	ivec4 screen_bounds = vec4(
		min(tiles_x, max(0, __float2int_rd((center.x - extent.x + 0.5f) / TILE_SIZE_PERSPCORRECT))), // x_min (inclusive)
		min(tiles_y, max(0, __float2int_rd((center.y - extent.y + 0.5f) / TILE_SIZE_PERSPCORRECT))), // y_min (inclusive)
		min(tiles_x, max(0, __float2int_ru((center.x + extent.x + 0.5f) / TILE_SIZE_PERSPCORRECT))), // x_max (exclusive)
		min(tiles_y, max(0, __float2int_ru((center.y + extent.y + 0.5f) / TILE_SIZE_PERSPCORRECT))) // y_max (exclusive)
	);

	// compute number of potentially influenced tiles
	int n_touched_tiles = (screen_bounds.z - screen_bounds.x) * (screen_bounds.w - screen_bounds.y);
	if (n_touched_tiles == 0) return;

	color = applyColorCorrection(color, colorCorrection);

	uint32_t flags = model.flags[splatIndex];

	{ // HANDLE FLAGS

		bool isSelected           = (flags & FLAGS_SELECTED) != 0;
		bool isDeleted            = (flags & FLAGS_DELETED) != 0;
		bool isHighlighted        = (flags & FLAGS_HIGHLIGHTED) != 0;
		bool isHighlightedNeg     = (flags & FLAGS_HIGHLIGHTED_NEGATIVE) != 0;

		auto rect = args.rectselect;
		if(isDeleted && isHighlighted && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			color.b = min(color.b * 2.0f + 0.4f, 1.0f);
			color.a = color.a / 4.0f;
		}else if(isDeleted){
			color.a = 0;
		}else if(isSelected && !isHighlightedNeg){
			color.r = min(color.r * 2.0f + 0.4f, 1.0f);
		}

		if(isHighlighted && (args.brush.mode == BRUSHMODE::ERASE)){
			color.r = min(color.r /  1.0f, 1.0f);
			color.g = min(color.g /  1.0f, 1.0f);
			color.b = min(color.b /  1.0f, 1.0f);
			color.a = min(color.a / 10.0f, 1.0f);
		}else if(isHighlighted && (args.brush.mode == BRUSHMODE::SELECT || rect.active)){
			color.r = min(color.r * 2.0f + 0.4f, 1.0f);
			color.g = min(color.g * 0.5f + 0.1f, 1.0f);
			color.b = min(color.b * 0.5f + 0.1f, 1.0f);
		}else if(isHighlighted && args.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			color.r = min(color.r * 0.5f + 0.1f, 1.0f);
			color.g = min(color.g * 0.5f + 0.1f, 1.0f);
			color.b = min(color.b * 2.0f + 0.4f, 1.0f);
		}
	}

	if(!model.writeDepth){
		flags = flags | FLAGS_DISABLE_DEPTHWRITE;
	}

	if((flags & FLAGS_DELETED) != 0) return;

	uint32_t visibleSplatID = atomicAdd(visibleSplatCounter, 1);
	ordering[visibleSplatID] = visibleSplatID;

	StageData_perspectivecorrect stuff;
	stuff.VPMT1 = VPMT1;
	stuff.VPMT2 = VPMT2;
	stuff.VPMT4 = VPMT4;
	stuff.MT3 = vec4(dot(M3, T[0]), dot(M3, T[1]), dot(M3, T[2]), z_view);
	stuff.color = to32BitColor(color);
	stuff.flags = flags;

	staging_bounds[visibleSplatID] = glm::i16vec4(screen_bounds);
	staging_data[visibleSplatID] = stuff;

	numTilefragments_splatwise[visibleSplatID] = n_touched_tiles;
	atomicAdd(numTilefragments, n_touched_tiles);

}

extern "C" __global__
void kernel_applyOrdering_u32(uint32_t* unsorted, uint32_t* sorted, uint32_t* ordering, uint32_t count){
	uint32_t index = cg::this_grid().thread_rank();

	if(index >= count) return;

	uint32_t order = ordering[index];
	sorted[index] = unsorted[order];
}

extern "C" __global__
void kernel_applyOrdering_xxx(uint8_t* unsorted, uint8_t* sorted, uint32_t* ordering, uint64_t stride, uint32_t count){
	uint64_t index = cg::this_grid().thread_rank();

	if(index >= count) return;

	uint64_t sourceOffset = uint64_t(ordering[index]) * stride;
	uint64_t targetOffset = index * stride;

	memcpy(sorted + targetOffset, unsorted + sourceOffset, stride);
}


extern "C" __global__
void kernel_applyOrdering_stagedata(StageData* unsorted, StageData* sorted, uint32_t* ordering, uint32_t count){
	uint32_t index = cg::this_grid().thread_rank();

	if(index >= count) return;

	sorted[index] = unsorted[ordering[index]];
}

extern "C" __global__
void kernel_createTilefragmentArray(
	// input
	CommonLaunchArgs args,
	RenderTarget target,
	uint32_t* ordering, 
	uint32_t numStagedSplats,
	StageData* stageDataArray,
	// uint32_t* dbg_numTilefragments_ordered,
	uint32_t* prefixsum,
	uint32_t tileFragmentsCounter,
	// output
	uint32_t* tileIDs,
	uint32_t* splatIDs
){
	
	// index of unsorted, staged splats
	uint32_t index = cg::this_grid().thread_rank();

	if(index >= numStagedSplats) return;

	// index of depth-sorted splats
	uint32_t order = ordering[index];

	// load stagedata of splats in depth-sorted order
	StageData stageData = stageDataArray[order];

	vec2 basisVector1, basisVector2;
	float depth;
	decode_stagedata(stageData, basisVector1, basisVector2, depth);

	vec2 pixelCoord = vec2(stageData.imgPos_encoded) / 10.0f;

	float quadHalfWidth  = sqrt(basisVector1.x * basisVector1.x + basisVector2.x * basisVector2.x);
	float quadHalfHeight = sqrt(basisVector1.y * basisVector1.y + basisVector2.y * basisVector2.y);

	float tiles_x = ceil(target.width / float(TILE_SIZE_3DGS));
	float tiles_y = ceil(target.height / float(TILE_SIZE_3DGS));

	ivec2 tile_start = {
		(pixelCoord.x - quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y - quadHalfHeight) / TILE_SIZE_3DGS};
	ivec2 tile_end = {
		(pixelCoord.x + quadHalfWidth) / TILE_SIZE_3DGS,
		(pixelCoord.y + quadHalfHeight) / TILE_SIZE_3DGS};

	tile_start.x = max(tile_start.x, 0);
	tile_end.x = min(tile_end.x, int(tiles_x) - 1);

	tile_start.y = max(tile_start.y, 0);
	tile_end.y = min(tile_end.y, int(tiles_y) - 1);

	int ltiles_x = (tile_end.x - tile_start.x) + 1;
	int ltiles_y = (tile_end.y - tile_start.y) + 1;
	int numTiles = ltiles_x * ltiles_y;

	uint32_t fragmentOffset = prefixsum[index];

	// forEachTouchedTile(pixelCoord, basisVector1, basisVector2, target, [&](uint32_t tile_x, uint32_t tile_y){
	forEachTouchedTile(args.uniforms.intersectionMode, pixelCoord, basisVector1, basisVector2, target, [&](uint32_t tile_x, uint32_t tile_y){
		uint32_t tileID = tile_x + tile_y * tiles_x;

		tileIDs[fragmentOffset] = tileID;
		splatIDs[fragmentOffset] = order;

		fragmentOffset++;
	});
	
}

extern "C" __global__
// void __launch_bounds__(256, 6) kernel_createTilefragmentArray_perspectivecorrect(
void kernel_createTilefragmentArray_perspectivecorrect(
	// input
	CommonLaunchArgs args,
	RenderTarget target,
	uint32_t numStagedSplats,
	glm::i16vec4* staging_bounds,
	uint32_t* prefixsum,
	uint32_t tileFragmentsCounter,
	// output
	uint32_t* tileIDs,
	uint32_t* splatIDs
){
	constexpr uint32_t block_size = 256u;
	constexpr uint32_t n_sequential_threshold = 2u;

	// index of unsorted, staged splats
	uint32_t primitive_idx = cg::this_grid().thread_rank();
	bool active = true;
	if (primitive_idx >= numStagedSplats) {
		active = false;
		primitive_idx = numStagedSplats - 1;
	}

	// load bounds of splats
	glm::i16vec4 screen_bounds_init = staging_bounds[primitive_idx];
	uint32_t tile_count_init = (screen_bounds_init.z - screen_bounds_init.x) * (screen_bounds_init.w - screen_bounds_init.y);

	if (tile_count_init == 0) active = false;
	if (__ballot_sync(0xffffffffu, active) == 0) return;

	uint32_t current_write_offset = prefixsum[primitive_idx];
	uint32_t screen_bounds_width_init = screen_bounds_init.z - screen_bounds_init.x;

	uint32_t tiles_x = (target.width + TILE_SIZE_PERSPCORRECT - 1) / TILE_SIZE_PERSPCORRECT;

	if (active) {
		for (uint32_t instance_idx = 0; instance_idx < tile_count_init && instance_idx < n_sequential_threshold; instance_idx++) {
			uint32_t tile_y = screen_bounds_init.y + (instance_idx / screen_bounds_width_init);
			uint32_t tile_x = screen_bounds_init.x + (instance_idx % screen_bounds_width_init);
			uint32_t tileID = tile_y * tiles_x + tile_x;

			tileIDs[current_write_offset] = tileID;
			splatIDs[current_write_offset] = primitive_idx;

			current_write_offset++;
		}
	}

	uint32_t thread_rank = cg::this_thread_block().thread_rank();
	uint32_t lane_idx = thread_rank % 32u;
	uint32_t warp_idx = thread_rank / 32u;
	uint32_t lane_mask_allprev_excl = 0xffffffffu >> (32u - lane_idx);

	const int compute_cooperatively = active && tile_count_init > n_sequential_threshold;
	const uint32_t remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);
	if (remaining_threads == 0) return;

	__shared__ glm::i16vec4 collected_screen_bounds[block_size];
	collected_screen_bounds[thread_rank] = screen_bounds_init;

	uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < 32; n++) {
		int i = __fns(remaining_threads, 0, n + 1); // find lane index of next remaining thread

		uint32_t primitive_idx_coop = __shfl_sync(0xffffffffu, primitive_idx, i);
		uint32_t current_write_offset_coop = __shfl_sync(0xffffffffu, current_write_offset, i);

		glm::i16vec4 screen_bounds = collected_screen_bounds[warp_idx * 32 + i];

		const uint32_t screen_bounds_width = screen_bounds.z - screen_bounds.x;
		const uint32_t tile_count = screen_bounds_width * (screen_bounds.w - screen_bounds.y);
		const uint32_t remaining_tile_count = tile_count - n_sequential_threshold;

		const int n_iterations = (remaining_tile_count + 31) / 32;
		for (int it = 0; it < n_iterations; it++) {
			const int instance_idx = it * 32 + lane_idx + n_sequential_threshold;
			const int active_curr_it = instance_idx < tile_count;

			uint32_t tile_y = screen_bounds.y + (instance_idx / screen_bounds_width);
			uint32_t tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);

			const uint32_t write = active_curr_it && true;

			const uint32_t write_ballot = __ballot_sync(0xffffffffu, write);
			const uint32_t n_writes = __popc(write_ballot);

			const uint32_t write_offset_it = __popc(write_ballot & lane_mask_allprev_excl);
			const uint32_t write_offset = current_write_offset_coop + write_offset_it;

			if (write) {
				uint32_t tileID = tile_y * tiles_x + tile_x;

				tileIDs[write_offset] = tileID;
				splatIDs[write_offset] = primitive_idx_coop;
			}
			current_write_offset_coop += n_writes;
		}
	}

}

extern "C" __global__
void kernel_prefilter_tiled_stagedata(
	Tile* tiles, 
	uint32_t* indices, 
	StageData* stagedatas
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int tileID = grid.block_rank();

	Tile tile = tiles[tileID];

	int numPointsInTile = clamp(tile.lastIndex - tile.firstIndex + 1, 0u, 1'000'000u);

	if(numPointsInTile < 10'000) return;

	int iterations = ((numPointsInTile + block.num_threads() - 1) / block.num_threads());

	for(int iteration = 0; iteration < iterations; iteration++){
		
		int index = block.num_threads() * iteration + block.thread_rank();
		
		if(index < numPointsInTile){
			int splatIndex = indices[tile.firstIndex + index];
			StageData stageData = stagedatas[splatIndex];

			uint32_t C = stageData.color;

			vec2 basisvector1, basisvector2;
			float depth;
			decode_stagedata(stageData, basisvector1, basisvector2, depth);

			float opacity = ((C >> 24) & 0xff) / 255.0f;

			// float lx = fmodf(data.imgpos.x, 16.0f);
			// float ly = fmodf(data.imgpos.y, 16.0f);
			// int lindex = int(lx) + 16 * int(ly);
			// bool hasLowOpacity = oapcity < 0.5f;
			// bool isSmall = length(basisvector1) < 10.0f && length(basisvector2) < 10.0f;
			// bool isRejected = isSmall && hasLowOpacity && sh_remainingTranslucency[lindex] <= 0.1f;
			// bool isAccepted = !isRejected;

			bool isAccepted = opacity > 0.002f;

			if(!isAccepted){
				stageData.color = stageData.color | 0xff0000ff;
				stagedatas[splatIndex] = stageData;
			}


		}
	}

	
}



// this method uses the same approach as INRIA / Kerbl & Kopanas et al.
// rasterizer_impl method identifyTileRanges()
// https://github.com/graphdeco-inria/gaussian-splatting
extern "C" __global__
void kernel_computeTiles_method1(
	CommonLaunchArgs args, RenderTarget target, 
	uint32_t* tileIDs, uint32_t numFragments,
	int tile_size,
	// output
	Tile* tiles
){
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index >= numFragments) return;

	int tiles_x = (target.width + tile_size - 1) / tile_size;
	int tiles_y = (target.height + tile_size - 1) / tile_size;
	uint32_t numTiles = tiles_x * tiles_y;


	uint32_t tileID = tileIDs[index];

	if(tileID >= numTiles) return;

	if(index == 0){
		tiles[tileID].firstIndex = 0;
	}else{
		uint32_t prevTileID = tileIDs[index - 1];

		if(tileID != prevTileID){
			tiles[prevTileID].lastIndex = index - 1;
			tiles[tileID].firstIndex = index;
		}
	}

	if(index == numFragments - 1){
		tiles[tileID].lastIndex = index;
	}
}

extern "C" __global__
void kernel_clearFramebuffer(CommonLaunchArgs args, RenderTarget target){

	int pixelID = cg::this_grid().thread_rank();
	int numPixels = target.width * target.height;

	if(pixelID >= numPixels) return;

	target.framebuffer[pixelID] = DEFAULT_PIXEL;
}

extern "C" __global__
void kernel_clearDepthbuffer(CommonLaunchArgs args, RenderTarget target){

	int pixelID = cg::this_grid().thread_rank();
	int numPixels = target.width * target.height;

	if(pixelID >= numPixels) return;

	Pixel* framebuffer = (Pixel*)target.framebuffer;

	Pixel pixel = framebuffer[pixelID];
	pixel.depth = Infinity;

	framebuffer[pixelID] = pixel;

	if(target.indexbuffer){
		target.indexbuffer[pixelID] = DEFAULT_PIXEL;
	}
}

extern "C" __global__
void kernel_toOpenGL(
	CommonLaunchArgs args,
	RenderTarget target,
	cudaSurfaceObject_t gl_colorbuffer
){
	auto grid = cg::this_grid();

	int width = target.width;
	int height = target.height;

	int pixelID = grid.thread_rank();
	int x = pixelID % width;
	int y = pixelID / width;

	if(pixelID >= width * height) return;

	uint64_t pixel = target.framebuffer[pixelID];
	uint32_t color = pixel & 0xffffffff;

	{ // show debug tile
		int tiles_x = (target.width + 16 - 1) / 16;
		int tile_x = x / 16;
		int tile_y = y / 16;
		int tileID = tile_x + tile_y * tiles_x;
		// if(isDebugTile(tile_x, tile_y))
		// if(x % 16 == 0 || x % 16 == 15 || y % 16 == 0 || y % 16 == 15)
		// {
		// 	color = 0xff0000ff;
		// }

		if(args.uniforms.inset.show){
			if(x == dbgtile_x * 16 + 16){
				color = 0xffff00ff;
			}
			if(x == dbgtile_x * 16 - 1){
				color = 0xffff00ff;
			}
			if(y == dbgtile_y * 16 - 1){
				color = 0xffff00ff;
			}
			if(y == dbgtile_y * 16 + 16){
				color = 0xffff00ff;
			}
		}
	}

	// color = 0xff0000ff;
	surf2Dwrite(color, gl_colorbuffer, x * 4, y);
}

extern "C" __global__
void kernel_blit_opengl(
	CommonLaunchArgs args,
	RenderTarget source,
	cudaSurfaceObject_t gl_colorbuffer,
	Rectangle target
){
	auto grid = cg::this_grid();

	int pixelID = grid.thread_rank();
	int target_relative_x = pixelID % int(target.width);
	int target_relative_y = pixelID / int(target.width);

	if(pixelID >= target.width * target.height) return;

	int target_x = target.x + target_relative_x;
	int target_y = target.y + target_relative_y;

	float u = float(target_relative_x + 0.5f) / float(target.width);
	float v = float(target_relative_y + 0.5f) / float(target.height);

	int source_x = clamp(float(source.width) * u, 0.0f, source.width - 1.0f);
	int source_y = clamp(float(source.height) * v, 0.0f, source.height - 1.0f);
	int source_pixelID = source_x + source_y * source.width;

	uint64_t sourcePixel = source.framebuffer[source_pixelID];
	uint32_t color = sourcePixel & 0xffffffff;

	surf2Dwrite(color, gl_colorbuffer, target_x * 4, target_y);
}

// extern "C" __global__
// void kernel_drawPoints_depth(CommonLaunchArgs args, GaussianData splats, RenderTarget target, uint32_t* depthbuffer, uint32_t* accumlatebuffer){

// 	auto grid = cg::this_grid();

// 	int index = grid.thread_rank();

// 	if(index >= splats.count) return;
	
// 	vec3 pos = splats.position[index];
// 	uint32_t flags = splats.flags[index];
// 	vec4 color = splats.color[index];
// 	float opacity = color.a;

// 	if(opacity < 10.0f) return;

// 	mat4 transform = target.proj * target.view * splats.transform;
// 	vec4 ndc = transform * vec4(pos, 1.0f);
// 	vec2 imgCoords = vec2(
// 		(0.5f * (ndc.x / ndc.w) + 0.5f) * target.width,
// 		(0.5f * (ndc.y / ndc.w) + 0.5f) * target.height
// 	);

// 	if(ndc.w < 0.0f) return;

// 	float size = 0.0f;

// 	for(float dx = -size; dx <= size; dx += 1.0f)
// 	for(float dy = -size; dy <= size; dy += 1.0f)
// 	{
// 		vec2 coord = imgCoords + vec2{dx, dy};

// 		if(coord.x < 0.0f || coord.x >= target.width) continue;
// 		if(coord.y < 0.0f || coord.y >= target.height) continue;

// 		int pixelID = int(coord.x) + int(coord.y) * target.width;

// 		uint32_t udepth = __float_as_uint(ndc.w);
// 		uint32_t oldDepth = depthbuffer[pixelID];

// 		if(udepth < oldDepth){
// 			atomicMin(&depthbuffer[pixelID], udepth);
// 		}
// 	}
// }

// extern "C" __global__
// void kernel_drawPoints_accumulate(CommonLaunchArgs args, GaussianData splats, RenderTarget target, float* depthbuffer, uint32_t* accumlatebuffer){

// 	auto grid = cg::this_grid();

// 	int index = grid.thread_rank();

// 	if(index >= splats.count) return;
	
// 	vec3 pos = splats.position[index];
// 	uint32_t flags = splats.flags[index];
// 	vec4 color = splats.color[index];
// 	// uint8_t* rgba = (uint8_t*)&color;
// 	float opacity = color.a;

// 	if(opacity < 1.0f) return;

// 	mat4 transform = target.proj * target.view * splats.transform;
// 	vec4 ndc = transform * vec4(pos, 1.0f);
// 	vec2 imgCoords = vec2(
// 		(0.5f * (ndc.x / ndc.w) + 0.5f) * target.width,
// 		(0.5f * (ndc.y / ndc.w) + 0.5f) * target.height
// 	);

// 	if(ndc.w < 0.0f) return;

// 	float size = 0.0f;

// 	for(float dx = -size; dx <= size; dx += 1.0f)
// 	for(float dy = -size; dy <= size; dy += 1.0f)
// 	{
// 		vec2 coord = imgCoords + vec2{dx, dy};

// 		if(coord.x < 0.0f || coord.x >= target.width) continue;
// 		if(coord.y < 0.0f || coord.y >= target.height) continue;

// 		int pixelID = int(coord.x) + int(coord.y) * target.width;

// 		uint32_t udepth = __float_as_uint(ndc.w);
// 		float oldDepth = depthbuffer[pixelID] * 1.01f;
// 		uint32_t uOldDepth = __float_as_int(oldDepth);


// 		if(udepth < uOldDepth){
// 			atomicAdd(&accumlatebuffer[4 * pixelID + 0], uint32_t(opacity * color.r * 255.0f));
// 			atomicAdd(&accumlatebuffer[4 * pixelID + 1], uint32_t(opacity * color.g * 255.0f));
// 			atomicAdd(&accumlatebuffer[4 * pixelID + 2], uint32_t(opacity * color.b * 255.0f));
// 			atomicAdd(&accumlatebuffer[4 * pixelID + 3], uint32_t(opacity * 255.0f));
// 		}
// 	}
// }

// extern "C" __global__
// void kernel_drawPoints_resolve(
// 	CommonLaunchArgs args, GaussianData splats, RenderTarget target, float* depthbuffer, 
// 	uint32_t* accumlatebuffer){

// 	auto grid = cg::this_grid();

// 	int index = grid.thread_rank();

// 	if(index >= target.width * target.height) return;
	
// 	int px = index % target.width;
// 	int py = index / target.width;
// 	int pixelID = px + py * target.width;

// 	int R = accumlatebuffer[4 * pixelID + 0];
// 	int G = accumlatebuffer[4 * pixelID + 1];
// 	int B = accumlatebuffer[4 * pixelID + 2];
// 	int count = accumlatebuffer[4 * pixelID + 3];

// 	int r = R / count;
// 	int g = G / count;
// 	int b = B / count;

// 	uint32_t color;
// 	uint8_t* rgba = (uint8_t*)&color;
// 	rgba[0] = r;
// 	rgba[1] = g;
// 	rgba[2] = b;

// 	float depth = depthbuffer[pixelID];
// 	uint64_t udepth = __float_as_uint(depth);

// 	uint64_t pixel = (udepth << 32) | color;
	
// 	uint64_t oldPixel = target.framebuffer[pixelID];
// 	uint32_t oldDepth = oldPixel >> 32;
// 	uint32_t oldColor = oldPixel & 0xffffffff;

// 	if(udepth < oldDepth){
// 		target.framebuffer[pixelID] = pixel;
// 	}
// }

template<typename T>
__device__ void swap(
    T& a,
    T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

struct FragmentInfo {
    float depth;
    uint32_t color;
    __device__ FragmentInfo(): depth(__FLT_MAX__), color(0) {}
    __device__ FragmentInfo(const float d, const vec4 c) : depth(d), color(to32BitColor(c)) {}
	__device__ vec4 get_color() {
		return vec4{
			(color >>  0) & 0xff,
			(color >>  8) & 0xff,
			(color >> 16) & 0xff,
			(color >> 24) & 0xff,
		} / 255.0f;
	}
};
__device__ __forceinline__ bool operator<(FragmentInfo& a, FragmentInfo& b) { return a.depth < b.depth; }

extern "C" __global__
void kernel_render_gaussians_perspectivecorrect(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData_perspectivecorrect* stagedatas
){
	auto grid     = cg::this_grid();
	auto block    = cg::this_thread_block();
	constexpr int BLOCK_SIZE = TILE_SIZE_PERSPCORRECT * TILE_SIZE_PERSPCORRECT;

	int width     = target.width;
	int height    = target.height;
	int tiles_x   = (width + TILE_SIZE_PERSPCORRECT - 1) / TILE_SIZE_PERSPCORRECT;
	int tiles_y   = (height + TILE_SIZE_PERSPCORRECT - 1) / TILE_SIZE_PERSPCORRECT;

	int tileID = grid.block_rank();
	
	Tile tile = tiles[tileID];

	// tile coordinates
	int tile_x = tileID % tiles_x;
	int tile_y = tileID / tiles_x;

	// this thread's pixel coordinate within tile
	int tilePixelIndex = threadIdx.x;
	int tile_pixel_x = tilePixelIndex % TILE_SIZE_PERSPCORRECT;
	int tile_pixel_y = tilePixelIndex / TILE_SIZE_PERSPCORRECT;

	// This thread's pixel coordinates within framebuffer
	int pixel_x = tile_x * TILE_SIZE_PERSPCORRECT + tile_pixel_x;
	int pixel_y = tile_y * TILE_SIZE_PERSPCORRECT + tile_pixel_y;
	float fpixel_x = __int2float_rn(pixel_x);
	float fpixel_y = __int2float_rn(pixel_y);
	int pixelID = pixel_x + pixel_y * width;

	bool inside = pixel_x < width && pixel_y < height;
	// setup shared memory
	__shared__ StageData_perspectivecorrect collected_stageData[BLOCK_SIZE];
	// initialize local storage
	float min_valid_core_depth = __FLT_MAX__;
	float transmittance_tail = 1.0f;
	vec4 rgba_premultiplied_tail = vec4(0.0f);
	FragmentInfo core_fragments[K];
	float oldDepth = inside ? __int_as_float(target.framebuffer[pixelID] >> 32) : Infinity;
	// collaborative loading and processing
	uint2 tile_range = make_uint2(tile.firstIndex, tile.lastIndex + 1);
	for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + tilePixelIndex; n_points_remaining > 0; n_points_remaining -= BLOCK_SIZE, current_fetch_idx += BLOCK_SIZE) {
		block.sync();
		if (current_fetch_idx < tile_range.y) collected_stageData[tilePixelIndex] = stagedatas[indices[current_fetch_idx]];
		block.sync();
		if (inside) {
			int current_batch_size = min(BLOCK_SIZE, n_points_remaining);
			for (int j = 0; j < current_batch_size; ++j) {
				StageData_perspectivecorrect stageData = collected_stageData[j];
				vec4 plane_x_diag = stageData.VPMT1 - stageData.VPMT4 * fpixel_x;
				vec4 plane_y_diag = stageData.VPMT2 - stageData.VPMT4 * fpixel_y;
				vec3 plane_x_diag_normal = vec3(plane_x_diag);
				vec3 plane_y_diag_normal = vec3(plane_y_diag);
				vec3 m = plane_x_diag.w * plane_y_diag_normal - plane_x_diag_normal * plane_y_diag.w;
				vec3 d = cross(plane_x_diag_normal, plane_y_diag_normal);
				float numerator_rho2 = dot(m, m);
				float denominator = dot(d, d);
				if (numerator_rho2 > MAX_CUTTOFF_SQ * denominator) continue; // considering opacity requires log/sqrt -> slower
				float denominator_rcp = 1.0f / denominator;
				float G = expf(-0.5f * numerator_rho2 * denominator_rcp);
				vec4 rgba = vec4{
					(stageData.color >>  0) & 0xff,
					(stageData.color >>  8) & 0xff,
					(stageData.color >> 16) & 0xff,
					(stageData.color >> 24) & 0xff,
				} / 255.0f;
				float alpha = fminf(rgba.w * G, MAX_FRAGMENT_ALPHA);
				if (alpha < MIN_ALPHA_THRESHOLD) continue;
				if (args.uniforms.showRing && alpha < 0.1f) alpha += 0.9f;

				vec3 eval_point_diag = cross(d, m) * denominator_rcp;
				float depth = dot(vec3(stageData.MT3), eval_point_diag) + stageData.MT3.w;
				if (depth >= oldDepth) continue;

				bool new_depth_valid = (stageData.flags & FLAGS_DISABLE_DEPTHWRITE) == 0;
				if (new_depth_valid && depth < min_valid_core_depth && alpha >= MIN_ALPHA_THRESHOLD_CORE) min_valid_core_depth = depth;

				vec4 rgba_premultiplied = vec4{
					rgba.r * alpha,
					rgba.g * alpha,
					rgba.b * alpha,
					alpha
				};
				if (depth < core_fragments[K - 1].depth && alpha >= MIN_ALPHA_THRESHOLD_CORE) {
					FragmentInfo core_fragment = FragmentInfo(depth, rgba_premultiplied);
					#pragma unroll
					for (int core_idx = 0; core_idx < K; ++core_idx)
						if (core_fragment < core_fragments[core_idx] && core_fragment.depth < __FLT_MAX__)
							swap(core_fragment, core_fragments[core_idx]);
					rgba_premultiplied = core_fragment.get_color();
				}
				rgba_premultiplied_tail += rgba_premultiplied;
				transmittance_tail *= 1.0f - rgba_premultiplied.a;
			}
		}
	}
	if (inside) {
		// blend core
		vec3 rgb_pixel = vec3(0.0f);
		float depth_pixel = min_valid_core_depth;
		float transmittance_core = 1.0f;
		bool done = false;
		#pragma unroll
		for (int core_idx = 0; core_idx < K && !done; ++core_idx) {
			vec4 rgba_premultiplied = core_fragments[core_idx].get_color();
			rgb_pixel += transmittance_core * vec3(rgba_premultiplied.x, rgba_premultiplied.y, rgba_premultiplied.z);
			transmittance_core *= 1.0f - rgba_premultiplied.w;
			if (transmittance_core < TRANSMITTANCE_THRESHOLD) done = true;
		}

		float transmittance_pixel = transmittance_core;
		// blend tail
		if (!done && rgba_premultiplied_tail.w >= MIN_ALPHA_THRESHOLD) {
			float weight_tail = transmittance_core * (1.0f - transmittance_tail);
			rgb_pixel += weight_tail * (1.0f / rgba_premultiplied_tail.w) * vec3(rgba_premultiplied_tail);
			transmittance_pixel *= transmittance_tail;
		}

		uint32_t color = 0;
		uint8_t* rgba = (uint8_t*)&color;
		if(args.uniforms.rendermode == RENDERMODE_COLOR){

			uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
			vec4 oldPixel = vec4{
				(C >>  0) & 0xff,
				(C >>  8) & 0xff,
				(C >> 16) & 0xff,
				(C >> 24) & 0xff,
			};

			rgba[0] = clamp(rgb_pixel.r * 255.0f + transmittance_pixel * oldPixel.r, 0.0f, 255.0f);
			rgba[1] = clamp(rgb_pixel.g * 255.0f + transmittance_pixel * oldPixel.g, 0.0f, 255.0f);
			rgba[2] = clamp(rgb_pixel.b * 255.0f + transmittance_pixel * oldPixel.b, 0.0f, 255.0f);
			rgba[3] = 255;

			uint32_t udepth = __float_as_uint(depth_pixel);
			uint64_t pixelValue = (uint64_t(udepth) << 32) | color;

			target.framebuffer[pixelID] = pixelValue;
		} else if(args.uniforms.rendermode == RENDERMODE_DEPTH){

			vec3 SPECTRAL[11] = {
				vec3{158,1,66},
				vec3{213,62,79},
				vec3{244,109,67},
				vec3{253,174,97},
				vec3{254,224,139},
				vec3{255,255,191},
				vec3{230,245,152},
				vec3{171,221,164},
				vec3{102,194,165},
				vec3{50,136,189},
				vec3{94,79,162},
			};

			float w = depth_pixel / 3.0f;
			float u = w - floor(w);

			int i0 = w;

			vec3 C0 = SPECTRAL[clamp((i0 + 0), 0, 10) % 11];
			vec3 C1 = SPECTRAL[clamp((i0 + 1), 0, 10) % 11];

			vec3 C = (1.0f - u) * C0 + u * C1;

			rgba[0] = C.x;
			rgba[1] = C.y;
			rgba[2] = C.z;

			uint32_t udepth = __float_as_uint(depth_pixel);

			target.framebuffer[pixelID] = (uint64_t(udepth) << 32) | color;
		}
	}

}


#include "./render/render_gaussians.cuh"
#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR
#include "./render/gaussian_selector_coverage.cuh"
#endif
#include "./render/render_gaussians_coretail.cuh"
#include "./render/render_gaussians_sortfree.cuh"
// #include "./render/render_gaussians_subsets.cuh"
#include "./render/render_gaussians_solid.cuh"
// #include "./render/render_gaussians_fetchfilter.cuh"
#include "./render/render_heatmap.cuh"
