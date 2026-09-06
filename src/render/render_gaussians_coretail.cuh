// Hybrid transparency for the Jacobian-projected 3DGS path.
// The tile list is already stable depth-ordered. Each pixel alpha-blends its
// first K sufficiently opaque splats and accumulates every other splat in an
// OIT tail.

constexpr float CORETTAIL_TRANSMITTANCE_THRESHOLD = 4.0f / 255.0f;
constexpr float CORETTAIL_TAIL_ALPHA_THRESHOLD = 1.0f / 255.0f;

extern "C" __global__
void kernel_render_gaussians_coretail(
	CommonLaunchArgs args, RenderTarget target,
	Tile* tiles, uint32_t* indices, StageData* stagedatas,
	uint32_t pointsInTileThreshold
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int width = target.width;
	int height = target.height;
	int tiles_x = int(width + TILE_SIZE_3DGS - 1) / int(TILE_SIZE_3DGS);
	int tileID = grid.block_rank();
	Tile tile = tiles[tileID];

	int tile_x = tileID % tiles_x;
	int tile_y = tileID / tiles_x;
	int tilePixelIndex = threadIdx.x;
	int tile_pixel_x = tilePixelIndex % TILE_SIZE_3DGS;
	int tile_pixel_y = tilePixelIndex / TILE_SIZE_3DGS;
	int pixel_x = tile_x * TILE_SIZE_3DGS + tile_pixel_x;
	int pixel_y = tile_y * TILE_SIZE_3DGS + tile_pixel_y;
	int pixelID = pixel_x + pixel_y * width;
	bool inside = pixel_x < width && pixel_y < height;

	int numPointsInTile = clamp(tile.lastIndex - tile.firstIndex + 1, 0u, 1'000'000u);
	__shared__ RasterizationData sh_rasterdata[PREFETCH_COUNT];

	// Keep the existing framebuffer depth as an occluder. Unlike the normal
	// renderer, the hybrid path does not update it while collecting the tail.
	float occlusionDepth = Infinity;
	if(inside){
		occlusionDepth = __int_as_float(target.framebuffer[pixelID] >> 32);
	}

	vec3 coreColor = vec3{0.0f};
	float coreTransmittance = 1.0f;
	int coreCount = 0;
	float firstCoreDepth = Infinity;
	uint32_t firstCoreFlags = 0;

	vec3 tailColorSum = vec3{0.0f};
	float tailAlphaSum = 0.0f;
	float tailTransmittance = 1.0f;
	float measurementDepth = __int_as_float(Infinity);
	float measurementTransmittance = 1.0f;
	float expectedDepthSum = 0.0f;
	float expectedWeightSum = 0.0f;
	float maxWeight = 0.0f;
	int coreLimit = clamp(args.uniforms.coreTailK, 1, 64);
	float coreAlphaThreshold = clamp(args.uniforms.coreTailCoreAlphaThreshold, 0.0f, 1.0f);
	float coreOpacity = clamp(args.uniforms.coreTailCoreOpacity, 0.0f, 1.0f);

	int iterations = (numPointsInTile + PREFETCH_COUNT - 1) / PREFETCH_COUNT;
	for(int iteration = 0; iteration < iterations; iteration++){
		int index = PREFETCH_COUNT * iteration + block.thread_rank();
		if(index < numPointsInTile && block.thread_rank() < PREFETCH_COUNT){
			int splatIndex = indices[tile.firstIndex + index];
			StageData stageData = stagedatas[splatIndex];
			uint32_t C = stageData.color;

			vec2 basisvector1, basisvector2;
			float depth;
			decode_stagedata(stageData, basisvector1, basisvector2, depth);

			RasterizationData data;
			data.dir1 = basisvector1 / dot(basisvector1, basisvector1);
			data.dir2 = basisvector2 / dot(basisvector2, basisvector2);
			data.imgpos = vec2(stageData.imgPos_encoded) / 10.0f;
			data.depth = depth;
			data.flags = stageData.flags;
			data.colors = vec4{
				(C >>  0) & 0xff,
				(C >>  8) & 0xff,
				(C >> 16) & 0xff,
				((C >> 24) & 0xff) / 255.0f,
			};
			sh_rasterdata[threadIdx.x] = data;
		}

		int splatsInSharedMemory = min(numPointsInTile - PREFETCH_COUNT * iteration, PREFETCH_COUNT);
		block.sync();

		if(inside){
			for(int i = 0; i < splatsInSharedMemory; i++){
				auto rasterdata = sh_rasterdata[i];
				if(rasterdata.depth > occlusionDepth) break;

				vec2 pFrag = {float(pixel_x) - rasterdata.imgpos.x, float(pixel_y) - rasterdata.imgpos.y};
				float sT = dot(rasterdata.dir1, pFrag);
				float sB = dot(rasterdata.dir2, pFrag);
				float w = sT * sT + sB * sB;
				if(w >= 1.0f) continue;

				vec4 splatColor = rasterdata.colors;
				w = exp(-4.0f * w);
				if(args.uniforms.showRing && w < 0.025f){
					splatColor.a = 1.0f;
					w += 0.5f;
				}

				float alpha = w * splatColor.a;
				if(alpha < CORETTAIL_TAIL_ALPHA_THRESHOLD) continue;

				float measurementWeight = measurementTransmittance * alpha;
				if(args.uniforms.measurementDepthMode == 3 &&
					pixel_x == int(args.mouseEvents.pos_x) && pixel_y == int(args.mouseEvents.pos_y)){
					args.state->measurementProfile.pixelX = pixel_x;
					args.state->measurementProfile.pixelY = pixel_y;
					uint32_t sampleIndex = atomicAdd(&args.state->measurementProfile.count, 1u);
					if(sampleIndex < MEASUREMENT_PROFILE_MAX_SAMPLES){
						args.state->measurementProfile.samples[sampleIndex] = {
							.depth = rasterdata.depth,
							.weight = measurementWeight,
						};
					}else{
						args.state->measurementProfile.overflow = 1;
					}
				}
				if(args.uniforms.measurementDepthMode == 0){
					if(measurementTransmittance * (1.0f - alpha) < args.uniforms.measurementTransmittanceThreshold && __float_as_uint(measurementDepth) == Infinity){
						measurementDepth = rasterdata.depth;
					}
				}else if(args.uniforms.measurementDepthMode == 1){
					expectedDepthSum += measurementWeight * rasterdata.depth;
					expectedWeightSum += measurementWeight;
				}else if((args.uniforms.measurementDepthMode == 2 || args.uniforms.measurementDepthMode == 3) && measurementWeight > maxWeight){
					maxWeight = measurementWeight;
					measurementDepth = rasterdata.depth;
				}
				measurementTransmittance *= 1.0f - alpha;

				if(coreCount < coreLimit && alpha >= coreAlphaThreshold){
					if(coreCount == 0){
						firstCoreDepth = rasterdata.depth;
						firstCoreFlags = rasterdata.flags;
					}
					// The slider changes only the exact front-to-back Core blend.
					// Tail splats still use their original alpha in the OIT accumulation below.
					float coreAlpha = alpha * coreOpacity;
					coreColor += coreTransmittance * coreAlpha * vec3(splatColor);
					coreTransmittance *= 1.0f - coreAlpha;
					coreCount++;
				}else if(coreTransmittance >= CORETTAIL_TRANSMITTANCE_THRESHOLD){
					tailColorSum += alpha * vec3(splatColor);
					tailAlphaSum += alpha;
					tailTransmittance *= 1.0f - alpha;
				}
			}
		}

		block.sync();
	}

	if(!inside) return;

	if(args.uniforms.measurementDepthMode == 1 && expectedWeightSum > 0.0f){
		measurementDepth = expectedDepthSum / expectedWeightSum;
	}
	if(target.measurementDepth && measurementDepth < target.measurementDepth[pixelID]){
		target.measurementDepth[pixelID] = measurementDepth;
	}

	float tailOpacity = 1.0f - tailTransmittance;
	vec3 tailColor = tailAlphaSum > 0.0f ? tailColorSum / tailAlphaSum : vec3{0.0f};
	vec3 pixelColor = coreColor + coreTransmittance * tailOpacity * tailColor;
	float remainingTranslucency = coreTransmittance * (1.0f - tailOpacity);

	if(args.uniforms.showTiles && (tile_pixel_x == 15 || tile_pixel_y == 15)){
		pixelColor *= 0.5f;
	}

	uint32_t C = target.framebuffer[pixelID] & 0xffffffff;
	vec3 oldColor = vec3{
		float((C >>  0) & 0xff),
		float((C >>  8) & 0xff),
		float((C >> 16) & 0xff),
	};

	uint32_t color = 0;
	uint8_t* rgba = (uint8_t*)&color;
	rgba[0] = clamp(pixelColor.r + remainingTranslucency * oldColor.r, 0.0f, 255.0f);
	rgba[1] = clamp(pixelColor.g + remainingTranslucency * oldColor.g, 0.0f, 255.0f);
	rgba[2] = clamp(pixelColor.b + remainingTranslucency * oldColor.b, 0.0f, 255.0f);
	rgba[3] = 255;

	uint64_t pixelValue = target.framebuffer[pixelID] & 0xffffffff'00000000;
	if(coreCount > 0 && remainingTranslucency < CORETTAIL_TRANSMITTANCE_THRESHOLD
		&& (firstCoreFlags & FLAGS_DISABLE_DEPTHWRITE) == 0){
		pixelValue = uint64_t(__float_as_uint(firstCoreDepth)) << 32;
	}
	target.framebuffer[pixelID] = pixelValue | color;
}
