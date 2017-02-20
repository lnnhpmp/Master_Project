
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride, uint N) {
	int index = get_global_id(0) * stride * 2;
	if ((index + stride < N) && index < N)
		array[index] += array[index + stride];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride, uint N) {
	int index = get_global_id(0);
	if (index + stride < N)
		array[index] += array[index + stride];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
	unsigned int GID = get_global_id(0);
	unsigned int LID = get_local_id(0);

	unsigned int res = 0;
	//compute the first reduction step
	while (GID < N) {
		res += inArray[GID];
		GID += get_global_size(0);
	}

	//load data into the local memory
	localBlock[LID] = res;
	barrier(CLK_LOCAL_MEM_FENCE);

	//do the reduction in local memory
	for (unsigned int stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
		if (LID < stride)
			localBlock[LID] += localBlock[LID + stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//read the result back to global memory
	if (LID == 0)
		outArray[get_group_id(0)] = localBlock[0];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
	uint GID = get_global_id(0);
	uint LID = get_local_id(0);
	//uint GSize = get_global_size(0);
	uint LSize = get_local_size(0);
	uint LSizeHalf = get_local_size(0) / 2;

	unsigned int res = 0;
	//compute the first reduction step
	while (GID < N) {
		res += inArray[GID];
		GID += get_global_size(0);
	}

	//load data into the local memory
	localBlock[LID] = res;
	barrier(CLK_LOCAL_MEM_FENCE);

	// local reduction
	for (uint stride = LSizeHalf; stride > 32; stride >>= 1) {
		if (LID < stride)
			localBlock[LID] += localBlock[LID + stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// unroll the loop
	uint warp = LSizeHalf > 32 ? 32 : LSizeHalf;
	if (LID < 32)
		for (uint stride = warp; stride > 1; stride >>= 1)
			localBlock[LID] += localBlock[LID + stride];
	// write the result back to global memory
	if (LID == 0)
		outArray[get_group_id(0)] = localBlock[0] + localBlock[1];
}

__kernel void Reduction_DecompCompleteUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
	uint GID = get_global_id(0);
	uint LID = get_local_id(0);
	//uint GSize = get_global_size(0);
	uint LSize = get_local_size(0);
	uint LSizeHalf = get_local_size(0) / 2;

	unsigned int res = 0;
	//compute the first reduction step
	while (GID < N) {
		res += inArray[GID];
		GID += get_global_size(0);
	}

	//load data into the local memory
	localBlock[LID] = res;
	barrier(CLK_LOCAL_MEM_FENCE);

	// local reduction
	for (uint stride = LSizeHalf; stride > 32; stride >>= 1) {
		if (LID < stride)
			localBlock[LID] += localBlock[LID + stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// unroll the loop
	uint warp = LSizeHalf > 32 ? 32 : LSizeHalf;
	if (LID < 32)
		for (uint stride = warp; stride > 1; stride >>= 1)
			localBlock[LID] += localBlock[LID + stride];
	// write the result back to global memory
	if (LID == 0)
		outArray[get_group_id(0)] = localBlock[0] + localBlock[1];
}
