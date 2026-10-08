function(ADD_IMGUI TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/imgui
		libs/imgui/backends)

	target_sources(${TARGET_NAME} PRIVATE
		libs/imgui/imgui.cpp
		libs/imgui/imgui_demo.cpp
		libs/imgui/imgui_draw.cpp
		libs/imgui/imgui_tables.cpp
		libs/imgui/imgui_widgets.cpp
		libs/imgui/backends/imgui_impl_glfw.cpp
		libs/imgui/backends/imgui_impl_opengl3.cpp)
endfunction()

function(ADD_IMPLOT TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/implot)
	target_sources(${TARGET_NAME} PRIVATE
		libs/implot/implot_items.cpp
		libs/implot/implot.cpp)
endfunction()

function(ADD_IMGUIZMO TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/ImGuizmo-1.83)
	target_sources(${TARGET_NAME} PRIVATE
		libs/ImGuizmo-1.83/ImGuizmo.cpp)
endfunction()



function(ADD_GLM TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE libs/glm)
endfunction()

if (WIN32)

	function(ADD_CUDA TARGET_NAME)
		find_package(CUDAToolkit 12.4 REQUIRED)
		target_include_directories(${TARGET_NAME} PRIVATE CUDAToolkit_INCLUDE_DIRS)
		cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH SPLATSHOP_CUDA_TOOLKIT_ROOT)
		target_compile_definitions(${TARGET_NAME} PRIVATE
			SPLATSHOP_CUDA_TOOLKIT_ROOT="${SPLATSHOP_CUDA_TOOLKIT_ROOT}")
		target_link_libraries(${TARGET_NAME}
			CUDA::cuda_driver
			CUDA::nvrtc
			CUDA::nvJitLink)

		# nvrtc and nvJitLink are shared libraries on Windows. Linking succeeds
		# without their DLLs beside the executable, but startup then fails with
		# STATUS_DLL_NOT_FOUND before main() can print a useful error.
		# CUDA's imported targets expose their .lib import libraries as
		# TARGET_FILE on Windows, so locate the versioned runtime DLLs in bin.
		# NVRTC also loads its matching builtins DLL dynamically.
		file(GLOB CUDA_RUNTIME_COMPILER_DLLS
			"${CUDAToolkit_BIN_DIR}/nvrtc64_*.dll"
			"${CUDAToolkit_BIN_DIR}/nvJitLink_*.dll"
			"${CUDAToolkit_BIN_DIR}/nvrtc-builtins64_*.dll")
		list(FILTER CUDA_RUNTIME_COMPILER_DLLS EXCLUDE REGEX "\\.alt\\.dll$")
		foreach(CUDA_RUNTIME_COMPILER_DLL IN LISTS CUDA_RUNTIME_COMPILER_DLLS)
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${CUDA_RUNTIME_COMPILER_DLL}"
					$<TARGET_FILE_DIR:${TARGET_NAME}>
				COMMENT "POST BUILD: copying CUDA runtime compiler DLL")
		endforeach()
	endfunction()

elseif (UNIX)

	function(ADD_CUDA TARGET_NAME)
		find_package(CUDAToolkit 12.4 REQUIRED)
		target_include_directories(${TARGET_NAME} PRIVATE CUDAToolkit_INCLUDE_DIRS)
		target_link_libraries(${TARGET_NAME}
			CUDA::cuda_driver
			CUDA::nvrtc
			${CUDAToolkit_LIBRARIES}
			${CUDAToolkit_CUDA_LIBRARY})
		find_library(NVJITLINK_LIBRARY nvJitLink REQUIRED HINTS $ENV{CUDA_PATH}/lib64/stubs ${CUDAToolkit_LIBRARY_DIRS})
		if (NVJITLINK_LIBRARY)
			target_link_libraries(${TARGET_NAME} ${NVJITLINK_LIBRARY})
		endif()
	endfunction()

endif (WIN32)

function(ADD_OPENGL TARGET_NAME)
	find_package(OpenGL REQUIRED)
	target_link_libraries(${TARGET_NAME} ${OPENGL_LIBRARY})

	target_include_directories(${TARGET_NAME} PRIVATE libs/glew/include)
	target_sources(${TARGET_NAME} PRIVATE libs/glew/glew.c)

	include(cmake/glfw.cmake)
	target_include_directories(${TARGET_NAME} PRIVATE ${glfw_SOURCE_DIR}/include)
	target_link_libraries(${TARGET_NAME} glfw)
endfunction()

# function(ADD_OPENVR TARGET_NAME)
# 
# 	find_package(OpenVR REQUIRED)
# 	target_link_libraries(main PRIVATE OpenVR::OpenVR)
# 
# #	target_include_directories(${TARGET_NAME} PRIVATE
# #		libs/openvr/headers)
# #
# #	target_link_libraries(${TARGET_NAME}
# #		CUDA::cuda_driver
# #		CUDA::nvrtc)
# #
# #	target_sources(${TARGET_NAME} PRIVATE
# #		libs/imgui/imgui.cpp
# #		libs/imgui/imgui_demo.cpp
# #		libs/imgui/imgui_draw.cpp
# #		libs/imgui/imgui_tables.cpp
# #		libs/imgui/imgui_widgets.cpp
# #		libs/imgui/backends/imgui_impl_glfw.cpp
# #		libs/imgui/backends/imgui_impl_opengl3.cpp)
# endfunction()
