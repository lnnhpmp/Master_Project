
__kernel void
set_array_to_constant(
	__global int *array,
	int num_elements,
	int val
)
{
	// There is no need to touch this kernel
	if(get_global_id(0) < num_elements)
		array[get_global_id(0)] = val;
}

__kernel void
compute_histogram(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins          // number of histogram bins
)
{
	// Insert your kernel code here
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	if (GID.x < width && GID.y < height) {
		float p = img[GID.y * pitch + GID.x] * (float)num_hist_bins;
		int h_idx = min(num_hist_bins - 1, max(0, (int)p));
		atomic_inc(&histogram[h_idx]);
	}
} 

__kernel void
compute_histogram_local_memory(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins,         // number of histogram bins
	__local int *local_hist
)
{
	// Insert your kernel code here
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	int thread_id = LID.y * get_local_size(0) + LID.x;

	// initialize local histogram
	if (thread_id < num_hist_bins)
		local_hist[thread_id] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (GID.x < width && GID.y < height) {
		float p = img[GID.y * pitch + GID.x] * (float)num_hist_bins;
		int h_idx = min(num_hist_bins - 1, max(0, (int)p));
		atomic_inc(&local_hist[h_idx]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// add the local histogram to the global histogram
	if (thread_id < num_hist_bins)
		atomic_add(&histogram[thread_id], local_hist[thread_id]);
} 
