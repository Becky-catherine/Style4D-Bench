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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define ENCODE_FREQ1 1.0f
#define ENCODE_FREQ2 2.0f
#define ENCODE_FREQ3 4.0f


/*__device__ void EncodeOneFloat2(float x, float* out, int& idx) {
    out[idx++] = sin(ENCODE_FREQ1 * x);
    out[idx++] = cos(ENCODE_FREQ1 * x);
}
*/
__device__ void EncodeOneFloat2(float x, float* out, int& idx) {
    out[idx++] = sin(ENCODE_FREQ1 * x);
    out[idx++] = cos(ENCODE_FREQ1 * x);
    out[idx++] = sin(ENCODE_FREQ2 * x);
    out[idx++] = cos(ENCODE_FREQ2 * x);
/*    out[idx++] = sin(ENCODE_FREQ3 * x);
    out[idx++] = cos(ENCODE_FREQ3 * x);
    */
}
/* define function */
__device__ glm::vec4 MySigmoid(glm::vec4 x)
{
	return glm::vec4(1.0f / (1.0f + exp(-x.x)), 1.0f / (1.0f + exp(-x.y)), 1.0f / (1.0f + exp(-x.z)), 1.0f / (1.0f + exp(-x.w)));
}

__device__ glm::vec4 dL_dSigmoid(glm::vec4 x)
{
	return glm::vec4(1.0f / (1.0f + exp(-x.x)) * (1.0f - 1.0f / (1.0f + exp(-x.x))),
					 1.0f / (1.0f + exp(-x.y)) * (1.0f - 1.0f / (1.0f + exp(-x.y))),
					 1.0f / (1.0f + exp(-x.z)) * (1.0f - 1.0f / (1.0f + exp(-x.z))),
					 1.0f / (1.0f + exp(-x.w)) * (1.0f - 1.0f / (1.0f + exp(-x.w))));
}
__device__ glm::vec4 MyRelu(glm::vec4 x)
{
	return glm::vec4(x.x > 0.0f? x.x : 0.0f,  x.y > 0.0f? x.y : 0.0f,  x.z > 0.0f? x.z : 0.0f,  x.w > 0.0f? x.w : 0.0f);
}
__device__ glm::vec4 dL_dRelu(glm::vec4 x)
{
	return glm::vec4(x.x > 0.0f? 1.0f : 0.0f,  x.y > 0.0f? 1.0f : 0.0f,  x.z > 0.0f? 1.0f : 0.0f,  x.w > 0.0f? 1.0f : 0.0f);
}
__device__ glm::vec4 MyTanh(glm::vec4 x)
{
	return glm::vec4(tanhf(x.x), tanhf(x.y), tanhf(x.z), tanhf(x.w));
}
__device__ glm::vec4 dL_dTanh(glm::vec4 x)
{
	// 先计算 tanh(x)
	float tx = tanhf(x.x);
	float ty = tanhf(x.y);
	float tz = tanhf(x.z);
	float tw = tanhf(x.w);
	
	// 再计算导数：1 - tanh2(x)
	return glm::vec4(1.0f - tx * tx,
					 1.0f - ty * ty,
					 1.0f - tz * tz,
					 1.0f - tw * tw);
}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * (36);   //max_coeffs; new 16+20

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * (36); //max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);

	// Compute loss gradient w.r.t. 3D means due to gradients of depth
	// from rendering procedure
	glm::vec3 dL_dmean2;
	float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepth[idx];
	dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepth[idx];
	dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepth[idx];

	// That's the third part of the mean gradient.
	dL_dmeans[idx] += dL_dmean2;
	
	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const glm::vec3* campos,  // new
	float focal_x, float focal_y,  // new
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float4* __restrict__ my_opacity,  // new
	const float* __restrict__ colors,
	const float* __restrict__ view2gaussian,  // new
	const float* __restrict__ time,  // new
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	const float* __restrict__ shs,  // new
	float* __restrict__ dL_dshs  // new
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	float2 ray = { float((pixf.x - W/2.) / focal_x), float((pixf.y - H/2.) / focal_y) };  // new

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float4 collected_my_opacity[BLOCK_SIZE];  // new
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 10];  // new
	__shared__ float collected_time[BLOCK_SIZE];  // new

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float dL_dpixel_depth;
	float accum_depth_rec = 0;
	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_my_opacity[block.thread_rank()] = my_opacity[coll_id];  // new
			collected_time[block.thread_rank()] = time[coll_id];  // new
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
			for (int ii = 0; ii < 10; ii++)    // new
				collected_view2gaussian[10 * block.thread_rank() + ii] = view2gaussian[coll_id * 10 + ii];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			float4 my_o = collected_my_opacity[j];  // new
			float my_time = collected_time[j];  // new
            /* start compute point */
            float* view2gaussian_j = collected_view2gaussian + j * 10;  // new
            float3 ray_point = make_float3(ray.x, ray.y, 1.0f);   // new
			const float normal[3] = {
				view2gaussian_j[0] * ray_point.x + view2gaussian_j[1] * ray_point.y + view2gaussian_j[2],
				view2gaussian_j[1] * ray_point.x + view2gaussian_j[3] * ray_point.y + view2gaussian_j[4],
				view2gaussian_j[2] * ray_point.x + view2gaussian_j[4] * ray_point.y + view2gaussian_j[5]
			};
			// use AA, BB, CC so that the name is unique
			double AA = ray_point.x * normal[0] + ray_point.y * normal[1] + normal[2];
			double BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
			float CC = view2gaussian_j[9];
			// t is the depth of the gaussian
			float t = -BB/(2*AA);
			// depth must be positive otherwise it is not valid and we skip it
			if (t <= NEAR_PLANE)
				continue;
      if (isnan(t))
        continue;
      if (isinf(t))
        continue;      
            // Must match the camera-space intersection used in forward.cu.
            float3 s = make_float3(
                t * ray_point.x,
                t * ray_point.y,
                t * ray_point.z
            );

			/* start MLP */
			float proj_u = s.x;
			float proj_v = s.y;
			float proj_c = s.z;
			float proj_t = my_time;
			glm::vec3* sh = ((glm::vec3*)shs) + collected_id[j] * (16+20);
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
            // forward
            //float input_encoded[4];
            //int cur_idx = 0;
            //EncodeOneFloat2(proj_u, input_encoded, cur_idx);
            //EncodeOneFloat2(proj_t, input_encoded, cur_idx);
            //glm::vec4 input_data2 = glm::vec4(input_encoded[0], input_encoded[1], input_encoded[2], input_encoded[3]);
            glm::vec4 input_data2 = glm::vec4(proj_u, proj_v, proj_c, proj_t);
            input_data2.x = tanh(proj_u);//1.0f / (1.0f + exp(-proj_u));
            input_data2.y = tanh(proj_v);//1.0f / (1.0f + exp(-proj_v));
            input_data2.z = tanh(proj_c);//1.0f / (1.0f + exp(-proj_c));
            input_data2.w = tanh(proj_t);//1.0f / (1.0f + exp(-proj_t));
            glm::vec4 L1out = input  * input_data2  + input_biases;   
            //glm::vec4 L1outR = MySigmoid(L1out);
            glm::vec4 L1outR = MyTanh(L1out);
			glm::vec4 L2out = middle * L1outR + middle_biases;
			//glm::vec4 L2outR = MySigmoid(L2out);
      glm::vec4 L2outR = MyTanh(L2out);
			glm::vec4 L3out = output * L2outR  + output_biases;
			//glm::vec4 L3outR = MySigmoid(L3out);
      glm::vec4 L3outR = MyTanh(L3out);
			float x[3] = {L3outR.x, L3outR.y, L3outR.z};
			const float raw_opa = my_o.x + 0.5f * L3outR.w;
			const bool opacity_active = raw_opa > 0.0f && raw_opa < 0.99f;
			float my_opa = min(0.99f, max(0.0f, raw_opa));



			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float raw_alpha = my_opa * G;
			const float alpha = min(0.99f, raw_alpha);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dpixel_depth_ddepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			// new
			float Myalpha = 1.0;
			float nncolor[3];
            nncolor[0] = x[0];
			nncolor[1] = x[1];
			nncolor[2] = x[2];
			float dl_dc[3];
			// compute color  decide whether to compute point loss
			//float dl_dc[3],dc_dsx[3]={0},dc_dsy[3],dc_dsz[3],dop_dsx=0.0,dop_dsy=0.0,dop_dsz=0.0;
			//float dsx_dtu[3],dsx_dtv[3],dsy_dtu[3],dsy_dtv[3],dsx_dtw[3],dsy_dtw[3];
            //  pending decision




			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j] + nncolor[ch];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				// new
				dl_dc[ch] = dchannel_dcolor * dL_dchannel;
			}
			const float c_d = collected_depths[j];
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_dpixel_depth;
			atomicAdd(&(dL_ddepths[global_id]), dpixel_depth_ddepth * dL_dpixel_depth);

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			const float effective_dL_dalpha = raw_alpha < 0.99f ? dL_dalpha : 0.0f;
      //const float lambda = 1e-6f;  // 可根据需要调整



			// Helpful reusable temporary variables
			const float dL_dG = my_opa * effective_dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			// new start compute tidu
			const float opacity_gradient = opacity_active ? G * effective_dL_dalpha : 0.0f;
			atomicAdd(&(dL_dopacity[global_id*4]), opacity_gradient);
            glm::vec4 dL_doutput_data = glm::vec4(
                dl_dc[0], dl_dc[1], dl_dc[2], 0.5f * opacity_gradient);
            dL_doutput_data = dL_doutput_data * dL_dTanh(L3out);
            glm::vec4 dL_dmiddle_data =  dL_doutput_data* output;
            glm::mat4x4 dL_doutput = glm::mat4x4(
				glm::vec4(L2outR.x*dL_doutput_data.x, L2outR.x*dL_doutput_data.y, L2outR.x*dL_doutput_data.z, L2outR.x*dL_doutput_data.w),
				glm::vec4(L2outR.y*dL_doutput_data.x, L2outR.y*dL_doutput_data.y, L2outR.y*dL_doutput_data.z, L2outR.y*dL_doutput_data.w),
				glm::vec4(L2outR.z*dL_doutput_data.x, L2outR.z*dL_doutput_data.y, L2outR.z*dL_doutput_data.z, L2outR.z*dL_doutput_data.w),
				glm::vec4(L2outR.w*dL_doutput_data.x, L2outR.w*dL_doutput_data.y, L2outR.w*dL_doutput_data.z, L2outR.w*dL_doutput_data.w)
			);
			glm::vec4 dL_doutput_biases = dL_doutput_data;
			dL_dmiddle_data= dL_dmiddle_data * dL_dTanh(L2out);
			glm::vec4 dL_dinput_data = dL_dmiddle_data * middle;
			glm::mat4x4 dL_dmiddle = glm::mat4x4(
				glm::vec4(L1outR.x*dL_dmiddle_data.x, L1outR.x*dL_dmiddle_data.y, L1outR.x*dL_dmiddle_data.z, L1outR.x*dL_dmiddle_data.w),
				glm::vec4(L1outR.y*dL_dmiddle_data.x, L1outR.y*dL_dmiddle_data.y, L1outR.y*dL_dmiddle_data.z, L1outR.y*dL_dmiddle_data.w),
				glm::vec4(L1outR.z*dL_dmiddle_data.x, L1outR.z*dL_dmiddle_data.y, L1outR.z*dL_dmiddle_data.z, L1outR.z*dL_dmiddle_data.w),
				glm::vec4(L1outR.w*dL_dmiddle_data.x, L1outR.w*dL_dmiddle_data.y, L1outR.w*dL_dmiddle_data.z, L1outR.w*dL_dmiddle_data.w)
			);
			glm::vec4 dL_dmiddle_biases = dL_dmiddle_data;
			dL_dinput_data = dL_dinput_data * dL_dTanh(L1out);
			glm::vec4 dL_dsxy = dL_dinput_data * input;
			glm::mat4x4 dL_dinput = glm::mat4x4(
				glm::vec4(input_data2.x*dL_dinput_data.x, input_data2.x*dL_dinput_data.y, input_data2.x*dL_dinput_data.z, input_data2.x*dL_dinput_data.w),
				glm::vec4(input_data2.y*dL_dinput_data.x, input_data2.y*dL_dinput_data.y, input_data2.y*dL_dinput_data.z, input_data2.y*dL_dinput_data.w),
				glm::vec4(input_data2.z*dL_dinput_data.x, input_data2.z*dL_dinput_data.y, input_data2.z*dL_dinput_data.z, input_data2.z*dL_dinput_data.w),
				glm::vec4(input_data2.w*dL_dinput_data.x, input_data2.w*dL_dinput_data.y, input_data2.w*dL_dinput_data.z, input_data2.w*dL_dinput_data.w)
			);
            glm::vec4 dL_dinput_biases = dL_dinput_data;
// 正则化
//dL_dinput += 2.0f * lambda * input;
//dL_dmiddle += 2.0f * lambda * middle;
//dL_doutput += 2.0f * lambda * output;
            atomicAdd(&dL_dshs[108*global_id + 48 ], dL_dinput[0].x);
			atomicAdd(&dL_dshs[108*global_id + 49 ], dL_dinput[0].y);
			atomicAdd(&dL_dshs[108*global_id + 50 ], dL_dinput[0].z);
			atomicAdd(&dL_dshs[108*global_id + 51 ], dL_dinput[0].w);
			atomicAdd(&dL_dshs[108*global_id + 52 ], dL_dinput[1].x);
			atomicAdd(&dL_dshs[108*global_id + 53 ], dL_dinput[1].y);
			atomicAdd(&dL_dshs[108*global_id + 54 ], dL_dinput[1].z);
			atomicAdd(&dL_dshs[108*global_id + 55 ], dL_dinput[1].w);
			atomicAdd(&dL_dshs[108*global_id + 56 ], dL_dinput[2].x);
			atomicAdd(&dL_dshs[108*global_id + 57 ], dL_dinput[2].y);
			atomicAdd(&dL_dshs[108*global_id + 58 ], dL_dinput[2].z);
			atomicAdd(&dL_dshs[108*global_id + 59 ], dL_dinput[2].w);

			atomicAdd(&dL_dshs[108*global_id + 104 ], dL_dinput[3].x);
			atomicAdd(&dL_dshs[108*global_id + 105 ], dL_dinput[3].y);
			atomicAdd(&dL_dshs[108*global_id + 106 ], dL_dinput[3].z);
			atomicAdd(&dL_dshs[108*global_id + 107 ], dL_dinput[3].w);

			atomicAdd(&dL_dshs[108*global_id + 60 ], dL_dinput_biases.x);
			atomicAdd(&dL_dshs[108*global_id + 61 ], dL_dinput_biases.y);
			atomicAdd(&dL_dshs[108*global_id + 62 ], dL_dinput_biases.z);
			atomicAdd(&dL_dshs[108*global_id + 63 ], dL_dinput_biases.w);

			atomicAdd(&dL_dshs[108*global_id + 64 ], dL_dmiddle[0].x);
			atomicAdd(&dL_dshs[108*global_id + 65 ], dL_dmiddle[0].y);
			atomicAdd(&dL_dshs[108*global_id + 66 ], dL_dmiddle[0].z);
			atomicAdd(&dL_dshs[108*global_id + 67 ], dL_dmiddle[0].w);
			atomicAdd(&dL_dshs[108*global_id + 68 ], dL_dmiddle[1].x);
			atomicAdd(&dL_dshs[108*global_id + 69 ], dL_dmiddle[1].y);
			atomicAdd(&dL_dshs[108*global_id + 70 ], dL_dmiddle[1].z);
			atomicAdd(&dL_dshs[108*global_id + 71 ], dL_dmiddle[1].w);
			atomicAdd(&dL_dshs[108*global_id + 72 ], dL_dmiddle[2].x);
			atomicAdd(&dL_dshs[108*global_id + 73 ], dL_dmiddle[2].y);
			atomicAdd(&dL_dshs[108*global_id + 74 ], dL_dmiddle[2].z);
			atomicAdd(&dL_dshs[108*global_id + 75 ], dL_dmiddle[2].w);
			atomicAdd(&dL_dshs[108*global_id + 76 ], dL_dmiddle[3].x);
			atomicAdd(&dL_dshs[108*global_id + 77 ], dL_dmiddle[3].y);
			atomicAdd(&dL_dshs[108*global_id + 78 ], dL_dmiddle[3].z);
			atomicAdd(&dL_dshs[108*global_id + 79 ], dL_dmiddle[3].w);
			atomicAdd(&dL_dshs[108*global_id + 80 ], dL_dmiddle_biases.x);
			atomicAdd(&dL_dshs[108*global_id + 81 ], dL_dmiddle_biases.y);
			atomicAdd(&dL_dshs[108*global_id + 82 ], dL_dmiddle_biases.z);
			atomicAdd(&dL_dshs[108*global_id + 83 ], dL_dmiddle_biases.w);

			atomicAdd(&dL_dshs[108*global_id + 84 ], dL_doutput[0].x);
			atomicAdd(&dL_dshs[108*global_id + 85 ], dL_doutput[0].y);
			atomicAdd(&dL_dshs[108*global_id + 86 ], dL_doutput[0].z);
			atomicAdd(&dL_dshs[108*global_id + 87 ], dL_doutput[0].w);
			atomicAdd(&dL_dshs[108*global_id + 88 ], dL_doutput[1].x);
			atomicAdd(&dL_dshs[108*global_id + 89 ], dL_doutput[1].y);
			atomicAdd(&dL_dshs[108*global_id + 90 ], dL_doutput[1].z);
			atomicAdd(&dL_dshs[108*global_id + 91 ], dL_doutput[1].w);
			atomicAdd(&dL_dshs[108*global_id + 92 ], dL_doutput[2].x);
			atomicAdd(&dL_dshs[108*global_id + 93 ], dL_doutput[2].y);
			atomicAdd(&dL_dshs[108*global_id + 94 ], dL_doutput[2].z);
			atomicAdd(&dL_dshs[108*global_id + 95 ], dL_doutput[2].w);
			atomicAdd(&dL_dshs[108*global_id + 96 ], dL_doutput[3].x);
			atomicAdd(&dL_dshs[108*global_id + 97 ], dL_doutput[3].y);
			atomicAdd(&dL_dshs[108*global_id + 98 ], dL_doutput[3].z);
			atomicAdd(&dL_dshs[108*global_id + 99 ], dL_doutput[3].w);
			atomicAdd(&dL_dshs[108*global_id + 100 ], dL_doutput_biases.x);
			atomicAdd(&dL_dshs[108*global_id + 101 ], dL_doutput_biases.y);
			atomicAdd(&dL_dshs[108*global_id + 102 ], dL_doutput_biases.z);
			atomicAdd(&dL_dshs[108*global_id + 103 ], dL_doutput_biases.w);

		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const glm::vec3* campos,  // new
	float focal_x, float focal_y,  // new
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float4* my_opacity,  // new
	const float* colors,
	const float* view2gaussian,  // new
	const float* time,  // new
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixel_depths,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths,
	const float* shs,  // new
    float* dL_dshs  // new
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		campos,  // new
		focal_x, focal_y,  // new
		bg_color,
		means2D,
		conic_opacity,
		my_opacity,  // new
		colors,
		view2gaussian,  // new
		time,  // new
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths,
		shs,  // new
        dL_dshs  // new
		);
}
