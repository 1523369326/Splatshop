// Sort-free weighted-sum rasterization for regular 3DGS StageData.
//
// The fragment list is deliberately kept unsorted.  A small atomic linked
// list is built for each screen tile, and the raster kernel accumulates every
// covered Gaussian using the commutative weighted average
//
//   C = sum_i(c_i * alpha_i * w(d_i)) / sum_i(alpha_i * w(d_i))
//
// with the fixed camera-depth weight w(d) defined in HostDeviceInterface.h.

extern "C" __global__
void kernel_createTilefragmentArray_sortfree(
	CommonLaunchArgs args,
	RenderTarget target,
	uint32_t numStagedSplats,
	StageData* stageDataArray,
	uint32_t* prefixsum,
	uint32_t* tileIDs,
	uint32_t* splatIDs
){
	uint32_t index = cg::this_grid().thread_rank();
	if(index >= numStagedSplats) return;

	StageData stageData = stageDataArray[index];
	vec2 basisVector1, basisVector2;
	float depth;
	decode_stagedata(stageData, basisVector1, basisVector2, depth);

	vec2 pixelCoord = vec2(stageData.imgPos_encoded) / 10.0f;
	uint32_t fragmentOffset = prefixsum[index];
	int tiles_x = (target.width + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;

	// This is the same conservative coverage test as the regular renderer,
	// but no ordering array is read: index is the unsorted staged splat ID.
	forEachTouchedTile(args.uniforms.intersectionMode, pixelCoord, basisVector1, basisVector2, target,
		[&](int tile_x, int tile_y){
			uint32_t tileID = uint32_t(tile_x + tile_y * tiles_x);
			tileIDs[fragmentOffset] = tileID;
			splatIDs[fragmentOffset] = index;
			fragmentOffset++;
		}
	);
}

extern "C" __global__
void kernel_build_sortfree_tile_lists(
	uint32_t* tileIDs,
	uint32_t* splatIDs,
	uint32_t numFragments,
	uint32_t numTiles,
	uint32_t* tileHeads,
	uint32_t* fragmentNext
){
	uint32_t fragmentIndex = cg::this_grid().thread_rank();
	if(fragmentIndex >= numFragments) return;

	uint32_t tileID = tileIDs[fragmentIndex];
	if(tileID >= numTiles){
		fragmentNext[fragmentIndex] = SORTFREE_INVALID_FRAGMENT;
		return;
	}

	// Atomic insertion is the only operation that establishes list order;
	// there is intentionally no depth or tile sort in this path.
	fragmentNext[fragmentIndex] = atomicExch(&tileHeads[tileID], fragmentIndex);
}

extern "C" __global__
void kernel_render_gaussians_sortfree(
	CommonLaunchArgs args,
	RenderTarget target,
	uint32_t* tileHeads,
	uint32_t* fragmentNext,
	uint32_t* splatIDs,
	StageData* stagedatas,
	uint32_t numVisibleSplats
){
	if(args.uniforms.splatRenderer != SPLATRENDERER_SORT_FREE) return;

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int width = target.width;
	int height = target.height;
	int tiles_x = (width + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;
	int tiles_y = (height + TILE_SIZE_3DGS - 1) / TILE_SIZE_3DGS;
	uint32_t numTiles = uint32_t(tiles_x * tiles_y);
	uint32_t tileID = grid.block_rank();
	if(tileID >= numTiles) return;

	int tile_x = int(tileID % uint32_t(tiles_x));
	int tile_y = int(tileID / uint32_t(tiles_x));
	int tile_pixel_x = int(threadIdx.x) % TILE_SIZE_3DGS;
	int tile_pixel_y = int(threadIdx.x) / TILE_SIZE_3DGS;
	int pixel_x = tile_x * TILE_SIZE_3DGS + tile_pixel_x;
	int pixel_y = tile_y * TILE_SIZE_3DGS + tile_pixel_y;
	int pixelID = pixel_x + pixel_y * width;
	bool inside = pixel_x < width && pixel_y < height;

	// One thread walks the unsorted linked list while the whole block evaluates
	// the loaded batch, matching the existing 16x16 tile execution shape.
	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];
	__shared__ uint32_t sh_splatIDs[PREFETCH_COUNT];
	__shared__ uint32_t sh_nextFragment;
	__shared__ uint32_t sh_batchCount;

	if(threadIdx.x == 0) sh_nextFragment = tileHeads[tileID];
	block.sync();

	float oldDepth = inside ? __int_as_float(target.framebuffer[pixelID] >> 32) : __int_as_float(Infinity);
	vec3 numerator = vec3{0.0f};
	float denominator = 0.0f;
	float nearestDepth = __int_as_float(Infinity);
	float nearestAnyDepth = __int_as_float(Infinity);
	float weightedDepthSum = 0.0f;
	float weightedDepthWeight = 0.0f;
	float maxContribution = 0.0f;
	float maxContributionDepth = __int_as_float(Infinity);

	while(true){
		if(threadIdx.x == 0){
			uint32_t fragment = sh_nextFragment;
			uint32_t count = 0;
			while(fragment != SORTFREE_INVALID_FRAGMENT && count < PREFETCH_COUNT){
				sh_splatIDs[count] = splatIDs[fragment];
				fragment = fragmentNext[fragment];
				count++;
			}
			sh_nextFragment = fragment;
			sh_batchCount = count;
		}
		block.sync();

		uint32_t batchCount = sh_batchCount;
		if(batchCount == 0) break;

		for(uint32_t i = uint32_t(threadIdx.x); i < batchCount; i += uint32_t(blockDim.x)){
			RasterizationData data;
			uint32_t splatIndex = sh_splatIDs[i];
			if(splatIndex < numVisibleSplats){
				StageData stageData = stagedatas[splatIndex];
				uint32_t C = stageData.color;
				vec2 basisVector1, basisVector2;
				float depth;
				decode_stagedata(stageData, basisVector1, basisVector2, depth);

				float basis1Length2 = dot(basisVector1, basisVector1);
				float basis2Length2 = dot(basisVector2, basisVector2);
				data.dir1 = basis1Length2 > 0.0f ? basisVector1 / basis1Length2 : vec2{0.0f};
				data.dir2 = basis2Length2 > 0.0f ? basisVector2 / basis2Length2 : vec2{0.0f};
				data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
				data.depth = depth;
				data.flags = stageData.flags;
				data.colors = vec4{
					float((C >>  0) & 0xff) / 255.0f,
					float((C >>  8) & 0xff) / 255.0f,
					float((C >> 16) & 0xff) / 255.0f,
					float((C >> 24) & 0xff) / 255.0f,
				};
				if(!(depth > 0.0f) || basis1Length2 <= 0.0f || basis2Length2 <= 0.0f){
					data.colors.a = 0.0f;
				}
			}else{
				data.dir1 = vec2{0.0f};
				data.dir2 = vec2{0.0f};
				data.imgpos = vec2{0.0f};
				data.depth = __int_as_float(Infinity);
				data.flags = FLAGS_DISABLE_DEPTHWRITE;
				data.colors = vec4{0.0f};
			}
			sh_rasterdata[i] = data;
		}
		block.sync();

		if(inside){
			for(uint32_t i = 0; i < batchCount; i++){
				RasterizationData rasterdata = sh_rasterdata[i];
				vec4 splatColor = rasterdata.colors;
				if(splatColor.a <= 0.0f) continue;
				if(rasterdata.depth >= oldDepth) continue;

				vec2 pFrag = vec2{float(pixel_x) - rasterdata.imgpos.x, float(pixel_y) - rasterdata.imgpos.y};
				float sT = dot(rasterdata.dir1, pFrag);
				float sB = dot(rasterdata.dir2, pFrag);
				float ellipseRadius2 = sT * sT + sB * sB;
				if(!(ellipseRadius2 < 1.0f)) continue;

				float gaussianWeight = expf(-4.0f * ellipseRadius2);
				if(args.uniforms.showRing && gaussianWeight < 0.025f){
					splatColor.a = 1.0f;
					gaussianWeight += 0.5f;
				}

				float alpha = fminf(gaussianWeight * splatColor.a, MAX_FRAGMENT_ALPHA);
				if(alpha < MIN_ALPHA_THRESHOLD) continue;

				float positiveDepth = fmaxf(rasterdata.depth, 0.0f);
				float depthWeight = expf(-SORTFREE_DEPTH_WEIGHT_SIGMA *
					powf(positiveDepth, SORTFREE_DEPTH_WEIGHT_BETA));
				float contribution = alpha * depthWeight;
				if(!(contribution > 0.0f)) continue;

				numerator += contribution * vec3(splatColor);
				denominator += contribution;
				weightedDepthSum += contribution * rasterdata.depth;
				weightedDepthWeight += contribution;
				if(rasterdata.depth < nearestAnyDepth) nearestAnyDepth = rasterdata.depth;
				if((rasterdata.flags & FLAGS_DISABLE_DEPTHWRITE) == 0 && rasterdata.depth < nearestDepth){
					nearestDepth = rasterdata.depth;
				}
				if(contribution > maxContribution){
					maxContribution = contribution;
					maxContributionDepth = rasterdata.depth;
				}

				if(args.uniforms.measurementDepthMode == 3 &&
					pixel_x == int(args.mouseEvents.pos_x) && pixel_y == int(args.mouseEvents.pos_y)){
					args.state->measurementProfile.pixelX = pixel_x;
					args.state->measurementProfile.pixelY = pixel_y;
					uint32_t sampleIndex = atomicAdd(&args.state->measurementProfile.count, 1u);
					if(sampleIndex < MEASUREMENT_PROFILE_MAX_SAMPLES){
						args.state->measurementProfile.samples[sampleIndex] = {
							.depth = rasterdata.depth,
							.weight = contribution,
						};
					}else{
						args.state->measurementProfile.overflow = 1;
					}
				}
			}
		}
		block.sync();
	}

	if(!inside) return;

	float measurementDepth = nearestAnyDepth;
	if(args.uniforms.measurementDepthMode == 1 && weightedDepthWeight > 0.0f){
		measurementDepth = weightedDepthSum / weightedDepthWeight;
	}else if(args.uniforms.measurementDepthMode == 2 || args.uniforms.measurementDepthMode == 3){
		measurementDepth = maxContributionDepth;
	}
	if(target.measurementDepth && measurementDepth < target.measurementDepth[pixelID]){
		target.measurementDepth[pixelID] = measurementDepth;
	}

	uint64_t oldPixel = target.framebuffer[pixelID];
	uint32_t oldColorPacked = uint32_t(oldPixel & 0xffffffffu);
	vec3 oldColor = vec3{
		float((oldColorPacked >>  0) & 0xff) / 255.0f,
		float((oldColorPacked >>  8) & 0xff) / 255.0f,
		float((oldColorPacked >> 16) & 0xff) / 255.0f,
	};
	vec3 pixelColor = denominator > 1e-8f ? numerator / denominator : oldColor;

	if(args.uniforms.showTiles && (tile_pixel_x == TILE_SIZE_3DGS - 1 || tile_pixel_y == TILE_SIZE_3DGS - 1)){
		pixelColor *= 0.5f;
	}

	if(args.uniforms.rendermode == RENDERMODE_DEPTH && weightedDepthWeight > 0.0f){
		float depthForColor = weightedDepthSum / weightedDepthWeight;
		float palettePosition = depthForColor / 3.0f;
		float u = palettePosition - floor(palettePosition);
		int i0 = int(palettePosition);
		vec3 spectrum[11] = {
			vec3{158,1,66}, vec3{213,62,79}, vec3{244,109,67},
			vec3{253,174,97}, vec3{254,224,139}, vec3{255,255,191},
			vec3{230,245,152}, vec3{171,221,164}, vec3{102,194,165},
			vec3{50,136,189}, vec3{94,79,162},
		};
		vec3 C0 = spectrum[clamp(i0, 0, 10) % 11];
		vec3 C1 = spectrum[clamp(i0 + 1, 0, 10) % 11];
		pixelColor = ((1.0f - u) * C0 + u * C1) / 255.0f;
	}

	uint32_t color = 0;
	uint8_t* rgba = (uint8_t*)&color;
	rgba[0] = clamp(pixelColor.r * 255.0f, 0.0f, 255.0f);
	rgba[1] = clamp(pixelColor.g * 255.0f, 0.0f, 255.0f);
	rgba[2] = clamp(pixelColor.b * 255.0f, 0.0f, 255.0f);
	rgba[3] = 255;

	uint64_t depthValue = oldPixel & 0xffffffff00000000ull;
	if(nearestDepth < oldDepth){
		depthValue = uint64_t(__float_as_uint(nearestDepth)) << 32;
	}
	target.framebuffer[pixelID] = depthValue | color;
}
