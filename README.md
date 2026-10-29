# Splatshop 

Paper: [Splatshop: Efficiently Editing Large Gaussian Splat Models](https://diglib.eg.org/bitstream/handle/10.1111/cgf70214/cgf70214.pdf), Markus Schütz, Christoph Peters, Florian Hahlbohm, Elmar Eisemann, Marcus Magnor, Michael Wimmer, Computer Graphics Forum 2025

This editor aims to enable editing of [gaussian splat models](https://github.com/graphdeco-inria/gaussian-splatting), e.g., cleaning up reconstructed models, assembling cleaned-up assets into a library, composing scenes, and painting. It currently supports up to a hundred million splats on desktop and up to around ten million in VR on an RTX 4090. 

There is still a lot of work to do - we are releasing it now in order to gather feedback and figure out which features and fixes to prioritize. 

An example data set is available [here](http://users.cg.tuwien.ac.at/mschuetz/permanent/splatmodels.zip). Unpack it, then drag&drop it into the editor. The garden splat model is from [Inria 3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and originates from [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9878829). 

A precompiled windows binary is available [here](http://users.cg.tuwien.ac.at/mschuetz/permanent/Splatshop_2025.03.21_v0.01f.zip). However, it may still be necessary to install CUDA 12.4 or later since CUDA kernels are compiled at startup. 

<img src="docs/paint_vr.gif" width="45%"/> <img src="docs/transform_vr.gif" width="45%"/>

### Features

<ul>
	<li> Undo and Redo
	<li> Virtual Reality viewing and editing
	<li> Select via brush, sphere, rectangle; Delete with brush or del button
	<li> Translate, scale, rotate of selected nodes or splats.
	<li> Painting
	<li> No pip or conda issues! (no Python)
	<li> CUDA Driver API - Can edit and hot-reload CUDA code like shaders.
	<li> Includes support for <a href="https://github.com/nerficg-project/HTGS">perspective-correct gaussians</a>. (Splat models need to be trained with respective method)
	<li> Viewer-side reproductions of Sort-Free weighted-sum rendering, Core-Tail hybrid transparency, and RaDe-GS geometry rendering.
	<li> Two-point measurement with four configurable depth-picking methods, including multi-surface CDF/PDF peak selection.
</ul>

### Research Reproductions and Experimental Rendering

This fork includes viewer-side reproductions and adaptations of several Gaussian-splatting papers. These implementations focus on interactive inference and visualization inside Splatshop; they do not reproduce the papers' complete training pipelines, loss functions, or benchmark results.

| Feature | Reference | Implementation in Splatshop | How to enable |
|---|---|---|---|
| **Sort-Free** | [Sort-Free Gaussian Splatting via Weighted Sum Rendering](https://arxiv.org/abs/2410.18931) (ICLR 2025) | Bypasses depth/fragment sorting and uses exponential depth-weighted sum rendering. The current viewer uses fixed weighting parameters and can be used to compare sort-free rendering on loaded models; reproducing the paper's full quality also requires its training procedure and learned view-dependent opacity. | In **Appearance**, select **Splat Renderer -> Sort-free**. |
| **Core-Tail** | [Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency](https://fhahlbohm.github.io/htgs/) (Eurographics 2025) | Reproduces the hybrid-transparency blending component on the regular 3DGS path: the first `K` sufficiently opaque fragments form an exactly alpha-blended core, while the remaining fragments are accumulated in an order-independent tail. | Select **Splat Renderer -> 3DGS**, then **Core-Tail**. Adjust **Core K**, **Core alpha cutoff**, and **Core blend opacity** in the controls. |
| **RaDe-GS Geometry** | [RaDe-GS: Rasterizing Depth in Gaussian Splatting](https://arxiv.org/abs/2406.01467) | Reproduces the ray-plane geometry evaluation used to obtain alpha-composited expected depth and camera-space normals from ordinary 3D Gaussians. | Enable **RaDe-GS geometry** in **Appearance**, open **Controls**, then choose **Color**, **Expected depth**, or **Camera-space normal**. |

The experimental renderers are selectable at runtime and CUDA kernels are compiled when the application starts. Some combinations use different intermediate data layouts, so enabling **RaDe-GS geometry** intentionally switches rendering to the compatible staged 3DGS path.

### Two-Point Measurement and Depth Picking

Click **Measure** in the toolbar, choose a depth-picking method, and press **Start measurement**. Pick two locations to display their Euclidean distance in world units. Measurements remain visible as an overlay and can be removed individually.

| Depth method | Description | Typical use |
|---|---|---|
| **First transmittance crossing** | Selects the first Gaussian after accumulated transmittance falls below the configurable `T` threshold. | Stable front-surface picking with an adjustable opacity threshold. |
| **Weighted expected depth** | Computes depth from alpha-contribution-weighted samples along the pixel ray. | Smooth depth estimates on opaque or softly blended surfaces. |
| **Maximum-weight Gaussian depth** | Uses the depth of the Gaussian with the largest alpha contribution. | Picking the dominant local surface without averaging foreground and background depths. |
| **CDF/PDF multi-surface depth** | Builds a per-ray depth distribution, smooths its PDF, detects multiple peaks, and lets the user select or drag to the intended surface before accepting each endpoint. This reproduces the multi-modal depth-distribution and peak-selection idea from [TSPE-GS: Probabilistic Depth Extraction for Semi-Transparent Surface Reconstruction via 3D Gaussian Splatting](https://arxiv.org/abs/2511.09944), adapted here for interactive measurement rather than the paper's full surface-reconstruction pipeline. | Disambiguating multiple visible surfaces, including geometry behind glass or other semi-transparent regions. |

For CDF/PDF picking, click a ray and wait for its GPU samples. The selector shows the PDF in yellow, CDF in blue, detected peaks in orange, and the selected depth in green. Choose a detected peak or drag the green marker, accept the point, and repeat for the second endpoint.

### Known Issues
<ul>
	<li> Academic Prototype - expect a fair amount of bugs and crashes -> Consider it an "alpha" version.
	<li> VR only on windows - No idea how to include OpenVR on linux. 
	<li> No Spherical Harmonics due to large mem req. with modest gains - we will check out more frugal SH options in the future. 
	<li> CUDA Driver API - Long startup because CUDA compiles at runtime. 
	<li> No precompiled binaries yet. 
	<li> Undo and redo may not cover all functionality yet. 
	<li> Undo of translate/scale/rotate is currently lossy. Especially scale is prone to loss of precision when scaling too far down. 
	<li> App freezing: Line rendering caused freezes on RTX 20xx series cards. That was fixed, but there is a chance that triangle models in VR mode may also freeze RTX 20xx series cards. 
</ul>

## Installing

<details>
<summary>Windows</summary>

Dependencies: 
* CUDA 12.4
* Visual Studio 2022 (version 17.10.3)
* CMake 3.27
* RTX 4070 or better recommended. 

Create Visual Studio solution files in a build folder via cmake:

```
mkdir build
cd build
cmake ../
```

Open build/SplatEditor.sln and compile in Release mode.<br>
Then run from project directory. 

```
./build/Release/SplatEditor.exe
```

Running from project directory is semi-important because the editor will look for resources such as ```./src/gaussians_rendering.cu``` and ```./resources/images/symbols_32_32.png``` relative to the project directory. Alternatively, you can copy the resources and src folders into the binary directory.



</details>

<details>
<summary>Linux</summary>

Dependencies:
* gcc 14, g++14
* CUDA 12.4 or higher
* NVIDIA driver version 550 or higher. (Lower may cause issues when compiling GPUSorting)

```
mkdir build
cd build
export CUDA_PATH=/usr/local/cuda-12.4/
cmake -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so ..
make
```

Potential Issues:
- May or may not require TBB. Uncomment the TBB part in CMakeLists, if it does.

```
// set CUDA_PATH variable to wherever your CUDA Toolkit is installed
export CUDA_PATH=/usr/local/cuda-12.4/

// May or may not require setting LD_LIBRARY_PATH to gcc 14, similar to this, or whereever gcc14 is installed
export LD_LIBRARY_PATH=~/gcc-14.1.0/x86_64-linux-gnu/libstdc++-v3/src/.libs:$LD_LIBRARY_PATH

// run from workspace root
./build/SplatEditor
```

</details>

### GaussianSelector (Windows)

GaussianSelector is integrated into the main editor and enabled by default.
To update an existing build that cached the feature as OFF, run from the repository root:

```powershell
cmake -S . -B build -DSPLATSHOP_ENABLE_GAUSSIAN_SELECTOR=ON -DSPLATSHOP_FETCH_GAUSSIAN_SELECTOR_DEPS=ON
cmake --build build --config Release --target SplatEditor -- /m
.\build\Release\SplatEditor.exe
```

The first configuration fetches pinned igraph sources unless a local source is supplied.
Click **Dev** next to **Dev&Debug**, then expand **GaussianSelector**.
Existing Sort-Free, Core-Tail and RaDe-GS controls remain in their original locations.
See [GaussianSelector usage](docs/gaussian_selector/usage.md) for ROI, strokes and selection history.

### Camera Studio

Open **Dev -> Camera Studio -> Open Camera Studio**. Two in-app panels appear: **Camera Trajectory** and **Camera Preview**; the existing main viewport remains the scene overview.

- **Create a path:** capture the current main view, insert cameras after the selection, click to add cameras on an XY/XZ/YZ plane, draw a stroke, or generate an orbit. The drawing canvas uses a sampled scene map; **Refresh map** updates it after scene changes. Middle drag pans the map and the mouse wheel zooms it.
- **Adjust cameras:** select a numbered marker in the map, list, or main scene. Drag map markers to change position, use the main scene's move/rotate gizmo (**T/R**), or edit position, rotation, vertical FOV, and segment duration numerically. Drag the selected camera preview to adjust its aim. **Edit selected in main view** switches the existing viewport to that camera: left drag aims, right drag pans, wheel/WASD move, and Q/E move vertically; **Back to overview** restores the overview camera.
- **Play:** use **Play path**, pause/stop, scrub the timeline, adjust speed, or repeat playback. **Follow path in main view** optionally drives the main camera; otherwise playback stays in the independent preview. Paths support centripetal Catmull-Rom position smoothing, quaternion rotation interpolation, and closed loops.
- **Save:** camera paths and output settings are saved to a separate JSON file, with path-specific undo/redo. This file is independent of the scene file.
- **Record:** **Record whole path** writes a silent **AVI/MJPEG** video using the existing JPEG encoder. Set resolution, FPS, quality, and output filename; frames advance deterministically even when rendering is slower than the chosen FPS. UI, trajectory markers, grid/axes, and Gaussian edit highlights are excluded. **Stop recording** finalizes the frames already written. Existing videos are protected unless overwrite is enabled; AVI recording stops before the single-file size limit.

Camera Studio uses the single desktop viewport. Opening it switches off Four Views and Stereo Framebuffer Test. Preview/video reuse the staged renderer (3DGS, Sort-Free, Core-Tail, perspective-correct, and RaDe-GS settings); the optional SoA/Bandwidth/Fragment Intersection layouts remain confined to the main viewport. Preview evaluates camera-dependent SH colors separately from the main camera's cache.

The feature adds no plugin, native window, or encoding program. Reconfigure CMake after adding the source files, then build as usual. New code comments are in concise Chinese.

## Getting Started

### Key and Mouse (as hardcoded in ```inputHandlingDesktop.h```)

<table>
	<tr>
		<td>Right Click</td>
		<td>Cancel current action</td>
	</tr>
	<tr>
		<td>Double Click</td>
		<td>Move towards hovered splats</td>
	</tr>
	<tr>
		<td>Ctrl + z, Ctrl + y</td>
		<td>Undo, Redo</td>
	</tr>
	<tr>
		<td>1</td>
		<td>Brush-Selection Action</td>
	</tr>
	<tr>
		<td>2</td>
		<td>Brush-Deletion Action</td>
	</tr>
	<tr>
		<td>t, r, s</td>
		<td>Translate, Rotate, Scale</td>
	</tr>
	<tr>
		<td>b</td>
		<td>Painting</td>
	</tr>
	<tr>
		<td>Ctrl + D</td>
		<td>Duplicate selection to new layer</td>
	</tr>
	<tr>
		<td>Ctrl + E</td>
		<td>Merge selected layer into the one below</td>
	</tr>
</table>

<img src="docs/vr_interaction.jpg" />

### Delete and Duplicate

<table>
	<tr>
		<td><img src="./docs/gardening_0.jpg" />1</td>
		<td><img src="./docs/gardening_1.jpg" />2</td>
		<td><img src="./docs/gardening_2.jpg" />3</td>
		</tr><tr>
		<td><img src="./docs/gardening_3.jpg" />4</td>
		<td><img src="./docs/gardening_4.jpg" />5</td>
		<td><img src="./docs/gardening_5.jpg" />6</td>
	</tr>
</table>

1. Select splats with spherical selection brush.
2. Delete selection (key: del).
3. Select a patch of grass with brush selection and intersection set to "center". This avoids selecting nearby large splats.
4. Duplicate selection (key: ctrl + d), then move it with translation tool (key: t). 
5. Repeat with varying patches of grass until the hole is covered. Overlapping lighter and darker patches may result in near-seamless transitions. 

### Cleanup

<table>
	<tr>
		<td><img src="./docs/cleanup_0.jpg" />1</td>
		<td><img src="./docs/cleanup_1.jpg" />2</td>
		<td><img src="./docs/cleanup_2.jpg" />3</td>
		<td colspan="2" rowspan="2"><img src="./docs/cleanup_6.jpg" width="1500px"/></td>
		</tr><tr>
		<td><img src="./docs/cleanup_3.jpg" />4</td>
		<td><img src="./docs/cleanup_4.jpg" />5</td>
		<td><img src="./docs/cleanup_5.jpg" />6</td>
		<td></td>
	</tr>
</table>

1. Drag&Drop ply file
2. Select region of interest with spherical selection tool. 
3. Invert selection and delete (key: del).
4. Rotate until model aligns with ground plane (key: r) and translate to origin (key: t). If the ground is sufficiently densely reconstructed, you can also try using the 3-point-alignment tool, which aligns the model by specifying three points on the ground. 
5. Remove undesired splats with a combination of circular and spherical selection and deletion tools. To retain a nice circular ground, select splats with a spherical brush
6. Then select the remainder of the splats that should remain. Invert selection and delete.

### Add an asset to library

<table>
	<tr>
		<td><img src="./docs/newasset_0.jpg" /></td>
		<td><img src="./docs/newasset_1.jpg" /></td>
		<td><img src="./docs/newasset_2.jpg" /></td>
		<td><img src="./docs/newasset_3.jpg" /></td>
	</tr>
</table>

1. Select splats
2. Duplicate selection (key: ctrl + d)
3. In the layers menu, right click the duplicated layer and give it a new name.
4. Right click the layer again, and select "Create asset from Layer".

### Adding new actions to toolbar

- Choose a suitable existing action and copy it, e.g. [BrushSelectAction](./src/actions/BrushSelectAction.h).
- Rename accordingly and adapt the code to your needs. Perhaps remove the undoable parts for now.
- ```update()``` is called every frame and handles most things. 
- ```makeToolbarSettings()``` allows you do specify an imgui user interface that is shown below the toolbar while your action is active. 
- Add your action to [toolbar.h](src/gui/toolbar.h). Include it and add a Button or ImageButton, similar to other actions. When the button is clicked, create a shared_ptr of your action and call setAction(action) to activate it. It will automatically be deactivated on right-click. 
- If you need to add an icon, you can add one to [symbols.svg](resources/images/symbols.svg) and export it to symbols_32x32.png. 

## Acknowledgements

* Contributors: Markus Schütz, Christoph Peters, Florian Hahlbohm, Elmar Eisemann, Michael Wimmer
* Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis for [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
* Mark Kellog for his WebGL implementation https://github.com/mkkellogg/GaussianSplats3D which helped substantially with the 3DGS implementation.
* Thomas Smith for his [GPUSorting](https://github.com/b0nes164/GPUSorting) project that is used to sort the gaussians in this project. 
* Omar's [Dear ImGui](https://github.com/ocornut/imgui). 
* This particular garden model is a pretrained model from [Inria 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and rotated to fit our editor. The garden data set originates from [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9878829).
* [SuperSplat](https://superspl.at/) - A web-based gaussian splat editor, from which we learned the usefullness of the ring rendering mode for editing, especially finding and removing floaters. 

## Citation

<pre>
@article{10.1111:cgf.70214,
	journal = {Computer Graphics Forum},
	title = {{Splatshop: Efficiently Editing Large Gaussian Splat Models}},
	author = {Schütz, Markus and Peters, Christoph and Hahlbohm, Florian and Eisemann, Elmar and Magnor, Marcus and Wimmer, Michael},
	year = {2025},
	publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
	ISSN = {1467-8659},
	DOI = {10.1111/cgf.70214}
}
</pre>
