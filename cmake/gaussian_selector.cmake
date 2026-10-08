set(GAUSSIAN_SELECTOR_CORE_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/CanonicalAxisColor.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/SpatialGraph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/LeidenAdapter.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/SuperpointEncoding.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/AdaptiveEncoding.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/SeedEvidence.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/SeedRefinement.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/AppearanceGMM.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/GraphCut.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/IntegerGraphCut.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/SelectorSession.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector/SelectorSelectionCommand.cpp)

add_library(gaussian_selector_core STATIC ${GAUSSIAN_SELECTOR_CORE_SOURCES})
target_include_directories(gaussian_selector_core PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian_selector)
target_compile_features(gaussian_selector_core PUBLIC cxx_std_23)

# The production encoding path must use igraph's weighted Leiden routine.  A
# missing dependency is an explicit configuration failure; there is no BFS or
# connected-components fallback masquerading as Leiden.
if(SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR)
	if(SPLATSHOP_FETCH_GAUSSIAN_SELECTOR_DEPS)
		include(FetchContent)
		set(IGRAPH_ENABLE_TLS OFF CACHE BOOL "" FORCE)
		set(IGRAPH_ENABLE_LTO OFF CACHE BOOL "" FORCE)
		set(IGRAPH_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
		set(IGRAPH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
		FetchContent_Declare(igraph
			URL https://github.com/igraph/igraph/releases/download/1.0.0/igraph-1.0.0.tar.gz
			URL_HASH SHA256=91e23e080634393dec4dfb02c2ae53ac4e3837172bb9047d32e39380b16c0bb0
			DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
		FetchContent_MakeAvailable(igraph)
	else()
		find_package(igraph 1.0 CONFIG REQUIRED)
	endif()
	if(TARGET igraph::igraph)
		target_link_libraries(gaussian_selector_core PRIVATE igraph::igraph)
	elseif(TARGET igraph)
		target_link_libraries(gaussian_selector_core PRIVATE igraph)
	else()
		message(FATAL_ERROR "igraph was configured but exported no supported CMake target")
	endif()
	target_compile_definitions(gaussian_selector_core PUBLIC GAUSSIAN_SELECTOR_HAS_IGRAPH=1)
endif()

if(SPLATSHOP_BUILD_GAUSSIAN_SELECTOR_TESTS)
	enable_testing()
	add_executable(gaussian_selector_tests
		${CMAKE_CURRENT_SOURCE_DIR}/tests/gaussian_selector/selector_tests.cpp)
	target_link_libraries(gaussian_selector_tests PRIVATE gaussian_selector_core)
	add_test(NAME gaussian_selector_core COMMAND gaussian_selector_tests)
endif()
