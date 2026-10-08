#pragma once

#include "HostDeviceInterface.h"
#include "CudaVirtualMemory.h"
#include "CURuntime.h"



struct GaussianDataManager{

	string name;

	shared_ptr<CudaVirtualMemory> vm_position            = nullptr;
	shared_ptr<CudaVirtualMemory> vm_scale               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_quaternion          = nullptr;
	shared_ptr<CudaVirtualMemory> vm_color               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_color_resolved      = nullptr;
	shared_ptr<CudaVirtualMemory> vm_depth               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_flags               = nullptr;
	shared_ptr<CudaVirtualMemory> vm_sphericalHarmonics  = nullptr;
	shared_ptr<CudaVirtualMemory> vm_cov3d             = nullptr;
	// shared_ptr<CudaVirtualMemory> vm_batches       = nullptr;

	GaussianData data;

	GaussianDataManager(string name){

		vm_position            = CURuntime::allocVirtual(format("[{}] position",   name)); 
		vm_scale               = CURuntime::allocVirtual(format("[{}] scale",      name)); 
		vm_quaternion          = CURuntime::allocVirtual(format("[{}] quaternion", name)); 
		vm_color               = CURuntime::allocVirtual(format("[{}] color",      name)); 
		vm_color_resolved      = CURuntime::allocVirtual(format("[{}] color resolved",      name)); 
		vm_depth               = CURuntime::allocVirtual(format("[{}] depth",      name)); 
		vm_flags               = CURuntime::allocVirtual(format("[{}] flags",      name)); 
		vm_sphericalHarmonics  = CURuntime::allocVirtual(format("[{}] spherical harmonics",name)); 
		vm_cov3d               = CURuntime::allocVirtual(format("[{}] cov3d", name));
		// vm_batches    = CURuntime::allocVirtual(format("[{}] batches",    name)); 

		data.position           = (decltype(data.position))    vm_position->cptr;
		data.scale              = (decltype(data.scale))       vm_scale->cptr;
		data.quaternion         = (decltype(data.quaternion))  vm_quaternion->cptr;
		data.color              = (decltype(data.color))       vm_color->cptr;
		data.color_resolved     = (decltype(data.color_resolved))   vm_color_resolved->cptr;
		data.depth              = (decltype(data.depth))       vm_depth->cptr;
		data.flags              = (decltype(data.flags))       vm_flags->cptr;
		data.sphericalHarmonics = (decltype(data.sphericalHarmonics))       vm_sphericalHarmonics->cptr;
		data.cov3d             = (decltype(data.cov3d)) vm_cov3d->cptr;
		// data.batches        = (decltype(data.batches))     vm_batches->cptr;

		this->name = name;
	}

	~GaussianDataManager(){
		// TODO: free memory...or not
	}

	void destroy(){

		CURuntime::free(vm_position);
		CURuntime::free(vm_scale);
		CURuntime::free(vm_quaternion);
		CURuntime::free(vm_color);
		CURuntime::free(vm_color_resolved);
		CURuntime::free(vm_depth);
		CURuntime::free(vm_flags);
		CURuntime::free(vm_sphericalHarmonics);
		CURuntime::free(vm_cov3d);
		// CURuntime::free(vm_batches);

		vm_position            = nullptr;
		vm_scale               = nullptr;
		vm_quaternion          = nullptr;
		vm_color               = nullptr;
		vm_color_resolved      = nullptr;
		vm_depth               = nullptr;
		vm_flags               = nullptr;
		vm_sphericalHarmonics  = nullptr;
		vm_cov3d              = nullptr;
		// vm_batches        = nullptr;

		data.count = 0;
	}

	// commit sufficient physical memory for the given amount of splats
	void commit(uint64_t numSplats){
		vm_position            ->commit(sizeof(*data.position  ) * numSplats, true);
		vm_scale               ->commit(sizeof(*data.scale     ) * numSplats, true);
		vm_quaternion          ->commit(sizeof(*data.quaternion) * numSplats, true);
		vm_color               ->commit(sizeof(*data.color     ) * numSplats, true);
		vm_color_resolved      ->commit(sizeof(*data.color_resolved) * numSplats, true);
		vm_depth               ->commit(sizeof(*data.depth     ) * numSplats, true);
		vm_flags               ->commit(sizeof(*data.flags     ) * numSplats, true);
		vm_sphericalHarmonics  ->commit(data.numSHCoefficients * sizeof(float) * numSplats, true);
		// Cov3D is committed lazily by Shared-MultiView. Keeping it virtual here
		// avoids an extra 24 bytes per Gaussian in the normal single-view path.

		// int numBatches = (numSplats + SPLATLET_SIZE - 1) / SPLATLET_SIZE;
		// vm_batches   ->commit(sizeof(*data.batches   ) * numBatches, true);

		// may need to update pointers, in case virtual address range changed.
		data.position            = (decltype(data.position))             vm_position->cptr;
		data.scale               = (decltype(data.scale))                vm_scale->cptr;
		data.quaternion          = (decltype(data.quaternion))           vm_quaternion->cptr;
		data.color               = (decltype(data.color))                vm_color->cptr;
		data.color_resolved      = (decltype(data.color_resolved))       vm_color_resolved->cptr;
		data.depth               = (decltype(data.depth))                vm_depth->cptr;
		data.flags               = (decltype(data.flags))                vm_flags->cptr;
		data.sphericalHarmonics  = (decltype(data.sphericalHarmonics))   vm_sphericalHarmonics->cptr;
		data.cov3d             = (decltype(data.cov3d)) vm_cov3d->cptr;
		// data.batches       = (decltype(data.batches))       vm_batches->cptr;
	}

	uint64_t getGpuMemoryUsage(){
		uint64_t usage = 0;
		
		usage += vm_position->comitted;
		usage += vm_scale->comitted;
		usage += vm_quaternion->comitted;
		usage += vm_color->comitted;
		usage += vm_color_resolved->comitted;
		usage += vm_depth->comitted;
		usage += vm_flags->comitted;
		usage += vm_sphericalHarmonics->comitted;
		usage += vm_cov3d->comitted;
		// usage += vm_batches->comitted;
		
		return usage;
	}

};

