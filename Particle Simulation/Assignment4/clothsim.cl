#define DAMPING 0.02f

#define G_ACCEL (float4)(0.f, -9.81f, 0.f, 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2	0.069f
#define WEIGHT_DIAG_2	0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f

// This is neccessary to scale the wind, otherwise it would be a class 10 Hurricane
#define WIND_COEFF (float4)(0.f, 0.f, 10.f, 0.f)

float dot3(float4 a, float4 b) 
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////
  __kernel void Integrate(unsigned int width,
			  unsigned int height, 
			  __global float4* d_pos,
			  __global float4* d_prevPos,
			  float elapsedTime,
			  float prevElapsedTime,
			  float simulationTime) {
							
    	int2 GID;
    	GID.x = get_global_id(0);
    	GID.y = get_global_id(1);
	
    	// Make sure the work-item does not map outside the cloth
	if(GID.x >= width || GID.y >= height)
		return;

	unsigned int particleID = GID.x + GID.y * width;
	// This is just to keep every 8th particle of the first row attached to the bar
    	if(particleID > width - 1 || (particleID & (7)) != 0) {
		// ADD YOUR CODE HERE!
		
		// Read the position
        	float4 pos = d_pos[particleID];
        	float4 pos_prev = d_prevPos[particleID];
		float dT = elapsedTime;
        	float dT_prev = prevElapsedTime;
	    	float omega = 0.5;	
        	
        	float4 pos_next;
		// Compute the new one using the Verlet position integration
		if (prevElapsedTime != 0.f)
			pos_next = pos + (pos - pos_prev) * dT / dT_prev + (G_ACCEL + WIND_COEFF * sin(omega * simulationTime)) * (dT + dT_prev) / 2.f * dT_prev;
		else
                       	pos_next = pos;
	

		// Move the value from d_pos into d_prevPos and store the new one in d_pos
        	d_pos[particleID] = pos_next;
        	d_prevPos[particleID] = pos;
        }
}



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// restDistance  - the distance between the given particles at rest
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
  float4 SatisfyConstraint(float4 pos1,
						 float4 pos2,
						 float restDistance){
	float4 toNeighbor = pos2 - pos1;
	return (toNeighbor - normalize(toNeighbor) * restDistance);
}

///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// restDistance     - the distance between two orthogonally neighboring particles at rest
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////

#define TILE_X 16 
#define TILE_Y 16
#define HALOSIZE 2

__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
__kernel void SatisfyConstraints(unsigned int width,
				 unsigned int height, 
				 float restDistance,
				 __global float4* d_posOut,
				 __global float4 const * d_posIn){
    
	// ADD YOUR CODE HERE!
	// Satisfy all the constraints (structural, shear, and bend).
	// You can use weights defined at the beginning of this file.

	// A ping-pong scheme is needed here, so read the values from d_posIn and store the results in d_posOut

	
    	// Allocate local memory with Halo
	__local float4 tile[TILE_X + 2 * HALOSIZE][TILE_Y + 2 * HALOSIZE];
	
    	int2 GID = (int2)(get_global_id(0), get_global_id(1));
    	int2 LID = (int2)(get_local_id(0), get_local_id(1));
    	int2 LS = (int2)(get_local_size(0), get_local_size(1));
	

    	if(GID.x >= width || GID.y >= height)
		return;

    	// Load Halo
    	// four edges
	// left 
	if (LID.x < HALOSIZE && GID.x >= HALOSIZE)
		tile[LID.x][LID.y + HALOSIZE] = d_posIn[GID.x - HALOSIZE + GID.y * width];

	// top 
	if (LID.y < HALOSIZE && GID.y >= HALOSIZE)
		tile[LID.x + HALOSIZE][LID.y] = d_posIn[GID.x + (GID.y - HALOSIZE) * width];

	// right 
	if (LID.x > LS.x - HALOSIZE - 1 && GID.x + HALOSIZE < width)
		tile[LID.x + HALOSIZE + 2][LID.y + HALOSIZE] = d_posIn[GID.x + HALOSIZE + GID.y * width];

	// bottom 
	if (LID.y > LS.y - HALOSIZE - 1 && GID.y + HALOSIZE < height)
		tile[LID.x + HALOSIZE][LID.y + HALOSIZE + 2] = d_posIn[GID.x + (GID.y + HALOSIZE) * width];

	// four corners
	// top left 
	if (LID.x < HALOSIZE && LID.y < HALOSIZE && GID.x >= HALOSIZE && GID.y >= HALOSIZE)
		tile[LID.x][LID.y] = d_posIn[GID.x - HALOSIZE + (GID.y - HALOSIZE) * width];

	//top right 
	if (LID.x > LS.x - HALOSIZE - 1 && LID.y < HALOSIZE && GID.x + HALOSIZE < width && GID.y >= HALOSIZE)
		tile[LID.x + HALOSIZE + 2][LID.y] = d_posIn[GID.x + HALOSIZE + (GID.y - HALOSIZE) * width];

	//bottom left 
	if (LID.x < HALOSIZE && LID.y > LS.y - HALOSIZE - 1 && GID.x >= HALOSIZE && GID.y + HALOSIZE < height)
		tile[LID.x][LID.y + HALOSIZE + 2] = d_posIn[GID.x - HALOSIZE + (GID.y + HALOSIZE) * width];

	//bottom right
	if (LID.x > LS.x - HALOSIZE - 1 && LID.y > LS.y - HALOSIZE - 1 && GID.x + HALOSIZE < width && GID.y + HALOSIZE < height)
		tile[LID.x + HALOSIZE + 2][LID.y + HALOSIZE + 2] = d_posIn[GID.x + HALOSIZE + (GID.y + HALOSIZE)*width];

	// load main data
	unsigned int particleID = GID.x + GID.y * width;
	
	tile[LID.x + HALOSIZE][LID.y + HALOSIZE] = d_posIn[particleID];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Hint: you should use the SatisfyConstraint helper function in the following manner:
	//SatisfyConstraint(pos, neighborpos, restDistance) * WEIGHT_XXX
	float4 res = d_posIn[particleID];
	
	if (particleID > width - 1 || (particleID & (7)) != 0)
	{
		// Structural constraints
        	// four neighbours 
        	// up
		if (GID.y > 0)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2][LID.y + 2 - 1], restDistance) * WEIGHT_ORTHO;
		// down
		if (GID.y < height - 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2][LID.y + 2 + 1], restDistance) * WEIGHT_ORTHO;
		// left
		if (GID.x > 0)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 - 1][LID.y + 2], restDistance) * WEIGHT_ORTHO;
		// right
		if (GID.x < width - 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 + 1][LID.y + 2], restDistance) * WEIGHT_ORTHO;


		// Shear constraints
        	// four neighbours 
		if (GID.y > 0 && GID.x > 0)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 - 1][LID.y + 2 - 1], restDistance*ROOT_OF_2) * WEIGHT_DIAG;

		if (GID.y > 0 && GID.x < width - 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2],tile[LID.x + 2 + 1][LID.y + 2 - 1],restDistance*ROOT_OF_2) * WEIGHT_DIAG;

		if (GID.y < height - 1 && GID.x > 0)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 - 1][LID.y + 2 + 1],restDistance*ROOT_OF_2) * WEIGHT_DIAG;

		if (GID.y < height - 1 && GID.x < width - 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 + 1][LID.y + 2 + 1], restDistance*ROOT_OF_2) * WEIGHT_DIAG;


		// Bend constraints
		if (GID.y > 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2][LID.y + 2 - 2], restDistance * 2) * WEIGHT_ORTHO_2;

		if (GID.y < height - 2)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2][LID.y + 2 + 2], restDistance * 2) * WEIGHT_ORTHO_2;

		if (GID.x > 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 - 2][LID.y + 2], restDistance * 2) * WEIGHT_ORTHO_2;

		if (GID.x < width - 2)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 + 2][LID.y + 2], restDistance * 2) * WEIGHT_ORTHO_2;

		if (GID.y > 1 && GID.x > 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 - 2][LID.y + 2 - 2], restDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;

		if (GID.y > 1 && GID.x < width - 2)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 + 2][LID.y + 2 - 2], restDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;

		if (GID.y < height - 2 && GID.x > 1)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 - 2][LID.y + 2 + 2], restDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;

		if (GID.y < height - 2 && GID.x < width - 2)
			res += SatisfyConstraint(tile[LID.x + 2][LID.y + 2], tile[LID.x + 2 + 2][LID.y + 2 + 2], restDistance*DOUBLE_ROOT_OF_2) * WEIGHT_DIAG_2;
	}
	
	// write back the data
	d_posOut[particleID] = res;
}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width,
			      unsigned int height, 
			      __global float4* d_pos,
			      float4 spherePos,
			      float sphereRad){								
	// ADD YOUR CODE HERE!
	// Find whether the particle is inside the sphere.
	// If so, push it outside.

	int2 GID = (int2)(get_global_id(0), get_global_id(1));
	unsigned int particleID = GID.x + GID.y * width;
	
	if(GID.x >= width || GID.y >= height)
		return;
    	
    	float4 diff = d_pos[particleID] - spherePos;
    	float distance = dot3(diff, diff);

    	// if collision
	if (distance < sphereRad * sphereRad)
		d_pos[particleID] = spherePos + normalize(diff) * sphereRad;

}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal( float4 p1, float4 p2, float4 p3) {
    float4 v1 = p2-p1;
    float4 v2 = p3-p1;

    return cross( v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								__global float4* d_normal){
								
    int particleID = get_global_id(0) + get_global_id(1) * width;
    float4 normal = (float4)( 0.0f, 0.0f, 0.0f, 0.0f);
    
    int minX, maxX, minY, maxY, cntX, cntY;
    minX = max( (int)(0), (int)(get_global_id(0)-1));
    maxX = min( (int)(width-1), (int)(get_global_id(0)+1));
    minY = max( (int)(0), (int)(get_global_id(1)-1));
    maxY = min( (int)(height-1), (int)(get_global_id(1)+1));
    
    for( cntX = minX; cntX < maxX; ++cntX) {
        for( cntY = minY; cntY < maxY; ++cntY) {
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY+1)],
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
        }
    }
    d_normal[particleID] = normalize( normal);
}
