
// Rotate the matrix CLOCKWISE

//naive implementation: move the elements of the matrix directly to their destinations
//this will cause unaligned memory accessed which - as we will see - should be avoided on the GPU

__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
	// TO DO: Add kernel code
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	if (GID.x < SizeX && GID.y < SizeY)
		MR[GID.x * SizeY + (SizeY - GID.y - 1)] = M[GID.y * SizeX + GID.x];
}

//this kernel does the same thing, however, the local memory is used to
//transform a small chunk of the matrix locally
//then write it back after synchronization in a coalesced access pattern

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR, uint SizeX, uint SizeY,
							__local float* block)
{
	// TO DO: Add kernel code
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	int2 LocalSize;
	LocalSize.x = get_local_size(0);
	LocalSize.y = get_local_size(1);

	// index of an element in local block
	int local_index = LID.y * LocalSize.x + LID.x;

	// load data into local memory.
	if (GID.x < SizeX && GID.y < SizeY)
		block[local_index] = M[GID.y * SizeX + GID.x];

	//we need to wait for other local threads to finish writing this shared array 
	barrier(CLK_LOCAL_MEM_FENCE);

	/*******now calculate the new LID and GID after rotating the tile matrix*******/
	//rotate the tile matrix
	int2 _LID;
	_LID.x = local_index / LocalSize.y;
	_LID.y = LocalSize.y - (local_index % LocalSize.y) - 1;

	//get the new global position
	int2 _GID;
	_GID.x = GID.x - LID.x + _LID.x;
	_GID.y = GID.y - LID.y + _LID.y;
	/*******now calculate the new LID and GID after rotating the tile matrix*******/

	// check the index
	if (_GID.x >= 0 && _GID.y >= 0 && _GID.x < SizeX && _GID.y < SizeY)
		// rotate the block and write it back
		MR[_GID.x * SizeY + (SizeY - _GID.y - 1)] = block[_LID.y * LocalSize.x + _LID.x];
}
 