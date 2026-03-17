// mi_cc_multivar.cpp  (v3 simplified)
//
// Multivariate MI: MI(X; Y) where X is d_x-dim, Y is d_y-dim.
//
// Pure brute-force implementation with true Chebyshev distance.
// No nanoflann dependency for distance computation — avoids the
// L1 vs Chebyshev radiusSearch issue entirely.
//
// Matches knnmi conventions exactly:
//   - RMS scale without centering (MutualInformationBase::scale)
//   - 1e-12 * mean * unif_rand() noise
//   - k+1 NN search including self
//   - nextafter(eps, 0) nudge
//   - radiusSearch-style counting: self included (dist 0 <= eps)
//   - digamma(count) where count includes self
//   - MI = psi(k) + psi(N) - <psi(n_x) + psi(n_y)>
//   - max(0, MI) clamp
//
// O(n^2) — at n=5000 in C++ this takes ~1-3 seconds.
//
// Copyright (C) 2026 — extends knnmi (GPL-3)

#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdint>

#include <Rmath.h>
#include "R.h"
#include "Rinternals.h"
#include <R_ext/Random.h>


// =============================================================================
// Helpers matching knnmi's MutualInformationBase exactly
// =============================================================================

// check_if_int: same logic as MutualInformationBase::check_if_int
static bool check_if_int(const double *col, int n) {
  double eps = std::numeric_limits<double>::epsilon();
  for (int i = 0; i < n; i++) {
    auto ival = static_cast<std::int64_t>(col[i]);
    if ((col[i] - ival) > eps) return false;
  }
  return true;
}

// scale_column: same logic as MutualInformationBase::scale
// RMS without centering, then 1e-12 * mean * U(0,1) noise
static void scale_column(double *col, int n, bool apply_scale) {
  if (apply_scale) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) sum_sq += col[i] * col[i];
    double rms = std::sqrt(sum_sq / (n - 1));
    if (rms > 0.0) {
      for (int i = 0; i < n; i++) col[i] /= rms;
    }
  }
  // Noise: 1e-12 * mean(scaled) * U(0,1)
  double mean_val = 0.0;
  for (int i = 0; i < n; i++) mean_val += col[i];
  mean_val /= n;
  for (int i = 0; i < n; i++) {
    col[i] += 1e-12 * mean_val * unif_rand();
  }
}

// Chebyshev distance between rows i and j of a column-major matrix
// mat is n x d stored as double[n*d] in column-major order
static inline double cheb_dist(const double *mat, int n, int d, int i, int j) {
  double maxd = 0.0;
  for (int dim = 0; dim < d; dim++) {
    double diff = std::fabs(mat[dim * n + i] - mat[dim * n + j]);
    if (diff > maxd) maxd = diff;
  }
  return maxd;
}


// =============================================================================
// Core implementation
// =============================================================================
static double mutual_inf_cc_mv_impl(
    const double *x_data, int n, int d_x,
    const double *y_data, int d_y,
    int k)
{
  int d_joint = d_x + d_y;
  
  // --- Allocate and populate working matrices (column-major: n x d) ---
  // x_scaled: n * d_x, y_scaled: n * d_y, joint: n * d_joint
  std::vector<double> x_scaled(n * d_x);
  std::vector<double> y_scaled(n * d_y);
  std::vector<double> joint(n * d_joint);
  
  // Copy X, scale each column
  for (int j = 0; j < d_x; j++) {
    for (int i = 0; i < n; i++) {
      x_scaled[j * n + i] = x_data[j * n + i];
    }
    bool is_int = check_if_int(&x_scaled[j * n], n);
    scale_column(&x_scaled[j * n], n, !is_int);
  }
  
  // Copy Y, scale each column
  for (int j = 0; j < d_y; j++) {
    for (int i = 0; i < n; i++) {
      y_scaled[j * n + i] = y_data[j * n + i];
    }
    bool is_int = check_if_int(&y_scaled[j * n], n);
    scale_column(&y_scaled[j * n], n, !is_int);
  }
  
  // Assemble joint = [X | Y] in column-major
  for (int j = 0; j < d_x; j++) {
    for (int i = 0; i < n; i++) {
      joint[j * n + i] = x_scaled[j * n + i];
    }
  }
  for (int j = 0; j < d_y; j++) {
    for (int i = 0; i < n; i++) {
      joint[(d_x + j) * n + i] = y_scaled[j * n + i];
    }
  }
  
  // --- Joint k-NN: find eps for each point ---
  // Brute-force O(n^2) with true Chebyshev distance
  std::vector<double> eps_vec(n);
  std::vector<double> all_dists(n);
  
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (j == i) {
        all_dists[j] = std::numeric_limits<double>::infinity();
      } else {
        all_dists[j] = cheb_dist(joint.data(), n, d_joint, i, j);
      }
    }
    // k-th nearest neighbor (0-indexed partial sort, k-1 for 0-based)
    std::nth_element(all_dists.begin(), all_dists.begin() + (k - 1),
                     all_dists.end());
    // Nudge down by 1 ULP (matching knnmi's nextafter trick)
    eps_vec[i] = std::nextafter(all_dists[k - 1], 0.0);
  }
  
  // --- Marginal counting with true Chebyshev distance ---
  double digamma_sum = 0.0;
  
  for (int i = 0; i < n; i++) {
    double eps_i = eps_vec[i];
    
    // X marginal count (includes self)
    double n_x = 1.0;  // self
    for (int j = 0; j < n; j++) {
      if (j == i) continue;
      double dx = cheb_dist(x_scaled.data(), n, d_x, i, j);
      if (dx <= eps_i) n_x += 1.0;
    }
    
    // Y marginal count (includes self)
    double n_y = 1.0;  // self
    for (int j = 0; j < n; j++) {
      if (j == i) continue;
      double dy = cheb_dist(y_scaled.data(), n, d_y, i, j);
      if (dy <= eps_i) n_y += 1.0;
    }
    
    // digamma(count) — count includes self, matching knnmi
    digamma_sum += Rf_digamma(n_x) + Rf_digamma(n_y);
  }
  
  // MI = psi(k) + psi(N) - <psi(n_x) + psi(n_y)>
  double mi = Rf_digamma((double)k) + Rf_digamma((double)n)
    - digamma_sum / (double)n;
  
  return std::max(0.0, mi);
}


// =============================================================================
// R-facing C wrapper
// =============================================================================
extern "C" {
  
  SEXP _mutual_inf_cc_mv(SEXP r_x, SEXP r_y, SEXP r_dx, SEXP r_dy, SEXP r_k) {
    int d_x = INTEGER(r_dx)[0];
    int d_y = INTEGER(r_dy)[0];
    int k_val = INTEGER(r_k)[0];
    int n = LENGTH(r_x) / d_x;
    
    GetRNGstate();
    
    SEXP mi = PROTECT(allocVector(REALSXP, 1));
    REAL(mi)[0] = mutual_inf_cc_mv_impl(
      REAL(r_x), n, d_x,
      REAL(r_y), d_y,
      k_val
    );
    
    PutRNGstate();
    UNPROTECT(1);
    return mi;
  }
  
} // extern "C"
