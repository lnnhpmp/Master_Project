


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) 
{
	// TO DO: Kernel implementation
	int GID = get_global_id(0);
	if (GID >= offset && GID < N)
		outArray[GID] = inArray[GID] + inArray[GID - offset];
	else
		outArray[GID] = inArray[GID];
}



// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
/*
There're 32 banks on modern GPUs
The bandwidth of shared memory is 32 bits per bank per clock cycle.
*/
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
#define OFFSET(A) ((A) + (A) / NUM_BANKS)
#else
#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock) 
{
	// TO DO: Kernel implementation
	// LocalID, number of work items in group and number of processed elements in group
	int LID = get_local_id(0);
	int sizeLocal = get_local_size(0);	
	int sizeBlock = sizeLocal * 2; // 2N elements to avoid poor thread utilization
	// Position of the element in the source data
	int GID = get_group_id(0) * sizeBlock + LID;

	// load data into local memory
	localBlock[OFFSET(LID)] = array[GID];
	localBlock[OFFSET(LID + sizeLocal)] = array[GID + sizeLocal];
	barrier(CLK_LOCAL_MEM_FENCE);

	// up sweep
	int stride;
	for (stride = 1; stride < sizeBlock; stride <<= 1) {
		// Position of the left element, it is shifted about [stride - 1] to the right to compensate,
		// that the elements are reduced towards the right side of the array
		int left = stride * (2 * LID + 1) - 1;
		int right = left + stride;
		if (right < sizeBlock)
			localBlock[OFFSET(right)] += localBlock[OFFSET(left)];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Set the last element int the block to zero
	if (LID == 0) 
		localBlock[OFFSET(sizeBlock - 1)] = 0;
	stride >>= 1;
	barrier(CLK_LOCAL_MEM_FENCE);

	// down sweep
	while (stride > 0) {
		// Right element of the down sweep
		// The LID gets inverted, so that the work item with the highest ID gets to write the element with the highest index.
		int right = sizeBlock - (sizeLocal - LID - 1) * stride * 2 - 1;
		int left = right - stride;
		// The way the right element is indexed, we only have to prevent work items not to touch elements with negative indices
		if (left >= 0) {
			uint tmp = localBlock[OFFSET(left)];
			localBlock[OFFSET(left)] = localBlock[OFFSET(right)];
			localBlock[OFFSET(right)] += tmp;
		}

		stride >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// Write back to main array and the higher level array
	array[GID] += localBlock[OFFSET(LID)];
	array[GID + sizeLocal] += localBlock[OFFSET(LID + sizeLocal)];
	// The work item that wrote the last element in the array, writes the one in the higher level too.
	if (LID == get_local_size(0) - 1) 
		higherLevelArray[get_group_id(0)] = array[GID + sizeLocal];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array)//, __local uint* localBlock) 
{
	// TO DO: Kernel implementation (large arrays)
	// Kernel that should add the group PPS to the local PPS (Figure 12)
	array[get_global_id(0) + get_local_size(0) * 2] += higherLevelArray[get_group_id(0) / 2];
}