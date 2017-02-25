/*
We assume a 3x3 (radius: 1) convolution kernel, which is not separable.
Each work-group will process a (TILE_X x TILE_Y) tile of the image.
For coalescing, TILE_X should be multiple of 16.

Instead of examining the image border for each kernel, we recommend to pad the image
to be the multiple of the given tile-size.
*/

//should be multiple of 32 on Fermi and 16 on pre-Fermi...
#define TILE_X 32 

#define TILE_Y 16

// d_Dst is the convolution of d_Src with the kernel c_Kernel
// c_Kernel is assumed to be a float[11] array of the 3x3 convolution constants, one multiplier (for normalization) and an offset (in this order!)
// With & Height are the image dimensions (should be multiple of the tile size)
__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
void Convolution(
				__global float* d_Dst,
				__global const float* d_Src,
				__constant float* c_Kernel,
				uint Width,  // Use width to check for image bounds
				uint Height,
				uint Pitch   // Use pitch for offsetting between lines
				)
{
	// OpenCL allows to allocate the local memory from 'inside' the kernel (without using the clSetKernelArg() call)
	// in a similar way to standard C.
	// the size of the local memory necessary for the convolution is the tile size + the halo area

	// local memory for the convolution + the halo area 
	__local float tile[TILE_Y + 2][TILE_X + 2];

	// TO DO...
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	// Fill the halo with zeros
	// the four edges (without corners)
	// left
	if (LID.x == 0)
		tile[LID.y + 1][0] = 0.f;

	// right 
	if (LID.x == TILE_X - 1)
		tile[LID.y + 1][TILE_X + 1] = 0.0f;

	// up
	if (LID.y == 0)
		tile[LID.y][LID.x + 1] = 0.0f;

	// low
	if (LID.y == TILE_Y - 1)
		tile[TILE_Y + 1][LID.x + 1] = 0.0f;

	// the four corners
	// top left
	if (LID.y == 0 && LID.x == 0)
		tile[0][0] = 0.0f;

	// top right
	if (LID.y == 0 && LID.x == TILE_X - 1)
		tile[0][TILE_X + 1] = 0.0f;

	// bottom left
	if (LID.y == TILE_Y - 1 && LID.x == 0)
		tile[TILE_Y + 1][0] = 0.0f;

	// bottom right
	if (LID.y == TILE_Y - 1 && LID.x == TILE_X - 1)
		tile[TILE_Y + 1][TILE_X + 1] = 0.0f;

	// Load main filtered area from d_Src

	// Load halo regions from d_Src (edges and corners separately), check for image bounds!

	// Sync threads


	// Fill the halo with zeros
	
	
	
	// Load main filtered area from d_Src
	if (GID.x < Width || GID.y < Height) {
		tile[LID.y + 1][LID.x + 1] = d_Src[GID.y * Pitch + GID.x];
	}

	// Load halo regions from d_Src (edges and corners separately), check for image bounds!
	bool left = LID.x < 1 && GID.x > 0;
	bool right = LID.x == TILE_X - 1 && GID.x + 1 < Width;
	bool top = LID.y < 1 && GID.y;
	bool bottom = LID.y == TILE_Y - 1 && GID.y + 1 < Height;
	bool top_left = top && left;
	bool top_right = top && right;
	bool bottom_left = bottom && left;
	bool bottom_right = bottom && right;

	// Left Vertical Edge
	if (left) {
		tile[LID.y + 1][0] = d_Src[GID.y * Pitch + GID.x - 1];
	}

	// Right Vertical Edge
	if (right) {
		tile[LID.y + 1][TILE_X + 1] = d_Src[GID.y * Pitch + GID.x + 1];
	}

	// Upper Horizontal Edge
	if (top) {
		tile[0][LID.x + 1] = d_Src[(GID.y - 1) * Pitch + GID.x];
	}

	// Lower Horizontal Edge
	if (bottom) {
		tile[TILE_Y + 1][LID.x + 1] = d_Src[(GID.y + 1) * Pitch + GID.x];
	}

	// Upper-Left Corner
	if (top_left) {
		tile[0][0] = d_Src[(GID.y - 1) * Pitch + GID.x - 1];
	}

	// Upper-Right Corner
	if (top_right) {
		tile[0][TILE_X + 1] = d_Src[(GID.y - 1) * Pitch + GID.x + 1];
	}

	// Lower-Left Corner
	if (bottom_left) {
		tile[TILE_Y + 1][0] = d_Src[(GID.y + 1) * Pitch + GID.x - 1];
	}

	// Lower-Right Corner
	if (bottom_right) {
		tile[TILE_Y + 1][TILE_X + 1] = d_Src[(GID.y + 1) * Pitch + GID.x + 1];
	}

	// Sync threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform the convolution and store the convolved signal to d_Dst.
	//apply convolution kernel
	float value = 0.0f;
	for (int offsetY = -1; offsetY < 2; offsetY++) {
		for (int offsetX = -1; offsetX < 2; offsetX++) {
			value += tile[LID.y + 1 + offsetY][LID.x + 1 + offsetX] * c_Kernel[(1 + offsetY) * 3 + (1 + offsetX)];
		}
	}

	d_Dst[GID.y * Pitch + GID.x] = value * c_Kernel[9] + c_Kernel[10];
}