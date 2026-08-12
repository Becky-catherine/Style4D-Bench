/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>



namespace cg = cooperative_groups;
#define ENCODE_FREQ1 1.0f
#define ENCODE_FREQ2 2.0f
#define ENCODE_FREQ3 4.0f
// NaN / Inf 检查宏
#define CHECK_NAN_INF(val, name) \
    if (isnan(val)) { printf("?? %s is NaN\n", name); } \
    else if (isinf(val)) { printf("?? %s is Inf\n", name); } 

// 检查 vec4 向量
#define CHECK_VEC4(v, name) \
    CHECK_NAN_INF(v.x, #name ".x") \
    CHECK_NAN_INF(v.y, #name ".y") \
    CHECK_NAN_INF(v.z, #name ".z") \
    CHECK_NAN_INF(v.w, #name ".w")
    
/*__device__ void EncodeOneFloat(float x, float* out, int& idx) {
    out[idx++] = sin(ENCODE_FREQ1 * x);
    out[idx++] = cos(ENCODE_FREQ1 * x);
}
*/

__device__ void EncodeOneFloat(float x, float* out, int& idx) {
    out[idx++] = sin(ENCODE_FREQ1 * x);
    out[idx++] = cos(ENCODE_FREQ1 * x);
    out[idx++] = sin(ENCODE_FREQ2 * x);
    out[idx++] = cos(ENCODE_FREQ2 * x);
/*    out[idx++] = sin(ENCODE_FREQ3 * x);
    out[idx++] = cos(ENCODE_FREQ3 * x);
    */
}

__device__ void check_values(glm::vec4 input_data2, glm::vec4 L1out, glm::vec4 L2out, glm::vec4 L3outR) {

    // 检查 proj 数据
    CHECK_NAN_INF(input_data2.x, "proj_u")
    CHECK_NAN_INF(input_data2.y, "proj_v")
    CHECK_NAN_INF(input_data2.z, "proj_c")
    CHECK_NAN_INF(input_data2.w, "proj_t")

    // 检查 MLP 输出
    CHECK_VEC4(L1out, "L1out")
    CHECK_VEC4(L2out, "L2out")
    CHECK_VEC4(L3outR, "L3outR")
}


// TODO combined with computeCov3D to avoid redundant computation
// Forward method for creating a view to gaussian coordinate system transformation matrix
__device__ void computeView2Gaussian(const glm::vec3 scale, const float3& mean, const glm::vec4 rot, const float* viewmatrix,  float* view2gaussian)
{
	// glm matrices use column-major order
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// Gaussian to world transform
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[1][0], R[2][0], 0.0f,
		R[0][1], R[1][1], R[2][1], 0.0f,
		R[0][2], R[1][2], R[2][2], 0.0f,
		mean.x, mean.y, mean.z, 1.0f
	);

	// could be simplied by using pointer
	// viewmatrix is the world to view transformation matrix
	glm::mat4 W2V = glm::mat4(
		viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	);

	// Gaussian to view transform
	glm::mat4 G2V = W2V * G2W;

	glm::mat3 R_transpose = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);

	glm::vec3 t = glm::vec3(G2V[3][0], G2V[3][1], G2V[3][2]);
	glm::vec3 t2 = -R_transpose * t;

    // precompute the value here to avoid repeated computations also reduce IO
	// v is the viewdirection and v^T is the transpose of v
	// t = position of the camera in the gaussian coordinate system
	// A = v^T @ R^T @ S^-1 @ S^-1 @ R @ v
	// B = t^T @ S^-1 @ S^-1 @ R @ v
	// C = t^T @ S^-1 @ S^-1 @ t
	// For the given caemra, t is fix and v depends on the pixel
	// therefore we can precompute A, B, C and use them in the forward pass
	// For A, we can precompute R^T @ S^-1 @ S^-1 @ R, which is a symmetric matrix and only store the upper triangle in 6 values
	// For B, we can precompute S^-1 @ S^-1 @ R @ v, which is a vector and store it in 3 values
	// and C is fixed, so we only need to store 1 value
	// Therefore, we only need to store 10 values in the view2gaussian matrix
	// S^-1 @ S^-1 is shared in A, B, C
	double3 S_inv_square = {1.0f / ((double)scale.x * scale.x + 1e-7), 1.0f / ((double)scale.y * scale.y + 1e-7), 1.0f / ((double)scale.z * scale.z + 1e-7)};
	double C = t2.x * t2.x * S_inv_square.x + t2.y * t2.y * S_inv_square.y + t2.z * t2.z * S_inv_square.z;
	glm::mat3 S_inv_square_R = glm::mat3(
		S_inv_square.x * R_transpose[0][0], S_inv_square.y * R_transpose[0][1], S_inv_square.z * R_transpose[0][2],
		S_inv_square.x * R_transpose[1][0], S_inv_square.y * R_transpose[1][1], S_inv_square.z * R_transpose[1][2],
		S_inv_square.x * R_transpose[2][0], S_inv_square.y * R_transpose[2][1], S_inv_square.z * R_transpose[2][2]
	);

	glm::vec3 B = t2 * S_inv_square_R;

	glm::mat3 Sigma = glm::transpose(R_transpose) * S_inv_square_R;

	// write to view2gaussian
	view2gaussian[0] = Sigma[0][0];
	view2gaussian[1] = Sigma[0][1];
	view2gaussian[2] = Sigma[0][2];
	view2gaussian[3] = Sigma[1][1];
	view2gaussian[4] = Sigma[1][2];
	view2gaussian[5] = Sigma[2][2];
	view2gaussian[6] = B.x;
	view2gaussian[7] = B.y;
	view2gaussian[8] = B.z;
	view2gaussian[9] = C;

}

/* new */
__device__ glm::vec4 MySigmoidf(glm::vec4 x)
{
	return glm::vec4(1.0f / (1.0f + exp(-x.x)), 1.0f / (1.0f + exp(-x.y)), 1.0f / (1.0f + exp(-x.z)), 1.0f / (1.0f + exp(-x.w)));
}

__device__ glm::vec4 MyReluf(glm::vec4 x)
{
	return glm::vec4(x.x > 0.0f? x.x : 0.0f,  x.y > 0.0f? x.y : 0.0f,  x.z > 0.0f? x.z : 0.0f,  x.w > 0.0f? x.w : 0.0f);
}
__device__ glm::vec4 MyTanhf(glm::vec4 x)
{
	return glm::vec4(tanhf(x.x), tanhf(x.y), tanhf(x.z), tanhf(x.w));
}
/* new */

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * (36); //max_coeffs; new
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* view2gaussian_precomp,  // new
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* view2gaussians,  // new
	float* rgb,
	float4* conic_opacity,
	float4* my_opacity,  // new
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx * 4] };   // new
	my_opacity[idx] = {opacities[idx*4 + 0], opacities[idx*4 + 1], opacities[idx*4 + 2], opacities[idx*4 + 3]};  // new
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

    // new 增加了计算view2gaussian部分
	const float* view2gaussian;
	if (view2gaussian_precomp == nullptr)
	{
		// printf("view2gaussian_precomp is nullptr\n");
		computeView2Gaussian(scales[idx], p_orig, rotations[idx], viewmatrix, view2gaussians + idx * 10);

	} else {
		view2gaussian = view2gaussian_precomp + idx * 16;
	}


}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const glm::vec3* __restrict__ cam_pos,  // new
	float focal_x, float focal_y,  // new
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ view2gaussian,  // new
	const float* __restrict__ time,  // new
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	const float4* __restrict__ my_opacity,  // new
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	const float* __restrict__ shs  // new
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// compute ray
	float2 ray = { float((pixf.x - W/2.) / focal_x), float((pixf.y - H/2.) / focal_y) };

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float4 collected_my_opacity[BLOCK_SIZE];  // new
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 10];  // new
	__shared__ float collected_time[BLOCK_SIZE];  // new

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_my_opacity[block.thread_rank()] = my_opacity[coll_id];  // new
			collected_time[block.thread_rank()] = time[coll_id];  // new
			for (int ii = 0; ii < 10; ii++)  // new
			    collected_view2gaussian[10 * block.thread_rank() + ii] = view2gaussian[coll_id * 10 + ii];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float4 my_o = collected_my_opacity[j];  // new
			float my_time = collected_time[j];  // new
			//start compute point
			float* view2gaussian_j = collected_view2gaussian + j * 10; // new
			float3 ray_point = make_float3(ray.x, ray.y, 1.0f);
            const float normal[3] = {
                    view2gaussian_j[0] * ray_point.x + view2gaussian_j[1] * ray_point.y + view2gaussian_j[2],
                    view2gaussian_j[1] * ray_point.x + view2gaussian_j[3] * ray_point.y + view2gaussian_j[4],
                    view2gaussian_j[2] * ray_point.x + view2gaussian_j[4] * ray_point.y + view2gaussian_j[5]
                };
            double AA = ray.x * normal[0] + ray.y * normal[1] + normal[2];
            double BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
			float CC = view2gaussian_j[9];
			float t = -BB/(2*AA);   // t is the depth of the gaussian
			if (t <= NEAR_PLANE)
				continue;
      if (isnan(t))
        continue;
      if (isinf(t))
        continue;
            // the scale of the gaussian is 1.f / sqrt(AA)
            //double min_value = -(BB/AA) * (BB/4.) + CC;
            // Intersection in camera coordinates. The old code mixed an
            // unrotated camera-space ray with a world-space camera center.
            float3 s = make_float3(
                t * ray_point.x,
                t * ray_point.y,
                t * ray_point.z
            );
      CHECK_NAN_INF(t, "t")



			/* 计算交点后的操作 */
			float proj_u = s.x;
			float proj_v = s.y;
			float proj_c = s.z;
      //printf("my_time: %f\n", s.x);
      //printf("my_time: %f\n", s.y);
      //printf("my_time: %f\n", s.z);
      //printf("end");
			float proj_t = my_time;
            glm::vec3* sh = ((glm::vec3*)shs) + collected_id[j] * (16+20);

            /* MLP construction */
			glm::mat4x4 input = glm::mat4x4(
				glm::vec4(sh[16+0].x, sh[16+0].y, sh[16+0].z, sh[16+1].x),
				glm::vec4(sh[16+1].y, sh[16+1].z, sh[16+2].x, sh[16+2].y),
				glm::vec4(sh[16+2].z, sh[16+3].x, sh[16+3].y, sh[16+3].z),
				glm::vec4(sh[16+18].z, sh[16+19].x, sh[16+19].y, sh[16+19].z)
			);
            glm::vec4 input_biases = glm::vec4(sh[16+4].x, sh[16+4].y, sh[16+4].z, sh[16+5].x);
            glm::mat4x4 middle = glm::mat4x4(
				glm::vec4(sh[16+5].y, sh[16+5].z, sh[16+6].x, sh[16+6].y),
				glm::vec4(sh[16+6].z, sh[16+7].x, sh[16+7].y, sh[16+7].z),
				glm::vec4(sh[16+8].x, sh[16+8].y, sh[16+8].z, sh[16+9].x),
				glm::vec4(sh[16+9].y, sh[16+9].z, sh[16+10].x, sh[16+10].y)
			);
			glm::vec4 middle_biases = glm::vec4(sh[16+10].z, sh[16+11].x, sh[16+11].y, sh[16+11].z);
            glm::mat4x4 output = glm::mat4x4(
				glm::vec4(sh[16+12].x, sh[16+12].y, sh[16+12].z, sh[16+13].x),
				glm::vec4(sh[16+13].y, sh[16+13].z, sh[16+14].x, sh[16+14].y),
				glm::vec4(sh[16+14].z, sh[16+15].x, sh[16+15].y, sh[16+15].z),
				glm::vec4(sh[16+16].x, sh[16+16].y, sh[16+16].z, sh[16+17].x)
			);
            glm::vec4 output_biases = glm::vec4(sh[16+17].y, sh[16+17].z, sh[16+18].x, sh[16+18].y);
            // input
            CHECK_NAN_INF(proj_u, "proj_u_before_compute")
            CHECK_NAN_INF(proj_v, "proj_v_before_compute")
            //CHECK_NAN_INF(proj_t, "proj_t_before_compute")
            //float input_encoded[4];
            //int cur_idx = 0;
            //EncodeOneFloat(proj_u, input_encoded, cur_idx);
            //EncodeOneFloat(proj_t, input_encoded, cur_idx);
            //glm::vec4 input_data2 = glm::vec4(input_encoded[0], input_encoded[1], input_encoded[2], input_encoded[3]);
            glm::vec4 input_data2 = glm::vec4(proj_u, proj_v, proj_c, proj_t);
            input_data2.x = tanh(proj_u);//1.0f / (1.0f + exp(-proj_u));
            input_data2.y = tanh(proj_v);//1.0f / (1.0f + exp(-proj_v));
            input_data2.z = tanh(proj_c);//1.0f / (1.0f + exp(-proj_c));
            input_data2.w = tanh(proj_t);
            // 3*4
            glm::vec4 L1out = input  * input_data2  + input_biases;
            //glm::vec4 L1outR = MySigmoidf(L1out);
            glm::vec4 L1outR = MyTanhf(L1out);
            
            // 4*4
            glm::vec4 L2out = middle * L1outR + middle_biases;
            //glm::vec4 L2outR = MySigmoidf(L2out);
            glm::vec4 L2outR = MyTanhf(L2out);
            // 4*4
            glm::vec4 L3out = output * L2outR  + output_biases;
            //glm::vec4 L3outR = MySigmoidf(L3out);
            glm::vec4 L3outR = MyTanhf(L3out);

            //float x[3] = {L3outR.x*2.0f -1.0f, L3outR.y*2.0f -1.0f, L3outR.z*2.0f -1.0f};
            //float my_opa = my_o.x + 0.5 * (L3outR.w * 2.0 - 1.0);
            float x[3] = {L3outR.x, L3outR.y, L3outR.z};
			float my_opa = min(0.99f, max(0.0f, my_o.x + 0.5f * L3outR.w));
            //check_values(input_data2, L1out, L2out, L3outR);





			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, my_opa * exp(power));  //con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

            // new compute color
            float Myalpha = 1.0;
            float nncolor[3];
            nncolor[0] = x[0];
            nncolor[1] = x[1];
            nncolor[2] = x[2];

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += (features[collected_id[j] * CHANNELS + ch] + Myalpha * nncolor[ch]) * alpha * T;
			D += depths[collected_id[j]] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const glm::vec3* cam_pos,  // new
	float focal_x, float focal_y,  // new
	const float2* means2D,
	const float* colors,
	const float* view2gaussian,  // new
	const float* time,  // new
	const float* depths,
	const float4* conic_opacity,
	const float4* my_opacity,  // new
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
	const float* shs  // new
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		cam_pos,  // new
		focal_x, focal_y,  // new
		means2D,
		colors,
		view2gaussian, // new
		time,  // new
		depths,
		conic_opacity,
		my_opacity,  // new
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		shs  // new
		);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* view2gaussian_precomp,  // new
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* view2gaussians, // new
	float* rgb,
	float4* conic_opacity,
	float4* my_opacity,  // new
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		view2gaussian_precomp,  // new
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		view2gaussians,  // new
		rgb,
		conic_opacity,
		my_opacity,  // new
		grid,
		tiles_touched,
		prefiltered
		);
}
