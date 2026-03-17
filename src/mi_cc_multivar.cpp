// mi_cc_multivar.cpp
//
// Multivariate mutual information: MI(X; Y) where X is d_x-dimensional
// and Y is d_y-dimensional.
//
// This extends knnmi's MutualInformation::compute() (which handles only
// the 1D vs 1D case) to arbitrary dimensions. Uses the exact same:
//   - nanoflann KD-tree with Chebyshev (L-infinity) metric
//   - nextafter eps-nudging trick
//   - radiusSearch for marginal counting (self-inclusive → digamma(count))
//   - RMS scaling without centering
//   - 1e-12 * mean * unif_rand() noise
//
// The only difference: joint space is (d_x + d_y)-dimensional, and
// marginal Chebyshev distances are computed over d_x or d_y dimensions
// respectively.
//
// Copyright (C) 2026 — extends knnmi (GPL-3)

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <limits>
#include <cstdint>

#include <Rmath.h>
#include "R.h"
#include "Rinternals.h"
#include <R_ext/Random.h>

#include "nanoflann.hpp"

// We need Eigen for the matrix adapter.
// knnmi bundles Eigen 3.4.0 in inst/include or similar.
// Adjust this include path if needed based on the actual knnmi layout.
#include <eigen3/Eigen/Dense>

using namespace Eigen;

// =============================================================================
// nanoflann adapter for dynamic-dimension Eigen matrices (Chebyshev metric)
// =============================================================================

// Chebyshev (L-infinity) distance metric for nanoflann
template <class T, class DataSource, typename _DistanceType = T>
struct ChebyshevDistance {
    typedef _DistanceType DistanceType;
    typedef T ElementType;

    const DataSource &data_source;

    ChebyshevDistance(const DataSource &_data_source) : data_source(_data_source) {}

    inline DistanceType evalMetric(const T *a, const size_t b_idx, size_t size) const {
        DistanceType result = DistanceType();
        for (size_t i = 0; i < size; ++i) {
            DistanceType diff = std::abs(a[i] - data_source.kdtree_get_pt(b_idx, i));
            if (diff > result) result = diff;
        }
        return result;
    }

    template <typename U, typename V>
    inline DistanceType accum_dist(const U a, const V b, const size_t) const {
        return std::abs(a - b);
    }
};

// Eigen matrix adapter for nanoflann (row-major, dynamic dims)
struct EigenMatrixAdaptor {
    typedef double coord_t;
    const MatrixXd &mat;

    EigenMatrixAdaptor(const MatrixXd &m) : mat(m) {}

    inline size_t kdtree_get_point_count() const { return mat.rows(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return mat(idx, dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    ChebyshevDistance<double, EigenMatrixAdaptor>,
    EigenMatrixAdaptor,
    -1  // dynamic dimensionality
> kd_tree_dynamic;


// =============================================================================
// Helper: RMS scale (matching knnmi's MutualInformationBase::scale)
// =============================================================================
static void scale_column(double *col, int n, bool apply_scale, bool add_noise) {
    if (apply_scale) {
        double sum_sq = 0.0;
        for (int i = 0; i < n; i++) sum_sq += col[i] * col[i];
        double rms = std::sqrt(sum_sq / (n - 1));
        if (rms > 0.0) {
            for (int i = 0; i < n; i++) col[i] /= rms;
        }
    }
    if (add_noise) {
        double mean_val = 0.0;
        for (int i = 0; i < n; i++) mean_val += col[i];
        mean_val /= n;
        GetRNGstate();
        for (int i = 0; i < n; i++) {
            col[i] += 1e-12 * mean_val * unif_rand();
        }
        PutRNGstate();
    }
}

// Check if a column is integer-valued (matching knnmi's check_if_int)
static bool check_if_int(const double *col, int n) {
    double eps = std::numeric_limits<double>::epsilon();
    for (int i = 0; i < n; i++) {
        int64_t ival = static_cast<int64_t>(col[i]);
        if (std::abs(col[i] - ival) > eps) return false;
    }
    return true;
}


// =============================================================================
// Core: mutual_inf_cc_mv_impl
//
// Computes MI(X; Y) where X is N x d_x and Y is N x d_y.
// Both X and Y are passed as column-major R matrices.
// =============================================================================
static double mutual_inf_cc_mv_impl(
    const double *x_data, int n, int d_x,
    const double *y_data, int d_y,
    int k)
{
    int d_joint = d_x + d_y;

    // --- Build joint matrix (N x d_joint) ---
    MatrixXd joint(n, d_joint);
    MatrixXd x_mat(n, d_x);
    MatrixXd y_mat(n, d_y);

    // Copy and scale X columns
    for (int j = 0; j < d_x; j++) {
        for (int i = 0; i < n; i++) {
            x_mat(i, j) = x_data[j * n + i];  // column-major from R
        }
        bool is_int = check_if_int(x_mat.col(j).data(), n);
        scale_column(x_mat.col(j).data(), n, !is_int, true);
    }

    // Copy and scale Y columns
    for (int j = 0; j < d_y; j++) {
        for (int i = 0; i < n; i++) {
            y_mat(i, j) = y_data[j * n + i];  // column-major from R
        }
        bool is_int = check_if_int(y_mat.col(j).data(), n);
        scale_column(y_mat.col(j).data(), n, !is_int, true);
    }

    // Assemble joint matrix
    joint.leftCols(d_x) = x_mat;
    joint.rightCols(d_y) = y_mat;

    // --- Build KD-tree for joint space ---
    EigenMatrixAdaptor joint_adaptor(joint);
    kd_tree_dynamic joint_tree(d_joint, joint_adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));
    joint_tree.buildIndex();

    // --- Build KD-trees for marginals ---
    EigenMatrixAdaptor x_adaptor(x_mat);
    kd_tree_dynamic x_tree(d_x, x_adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));
    x_tree.buildIndex();

    EigenMatrixAdaptor y_adaptor(y_mat);
    kd_tree_dynamic y_tree(d_y, y_adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(10));
    y_tree.buildIndex();

    // --- k-NN search in joint space ---
    int real_k = k + 1;  // +1 to include self (distance 0)
    std::vector<double> eps_vec(n);

    for (int i = 0; i < n; i++) {
        std::vector<uint32_t> ret_idx(real_k);
        std::vector<double> ret_dist(real_k);

        // Query point
        std::vector<double> query(d_joint);
        for (int j = 0; j < d_joint; j++) query[j] = joint(i, j);

        size_t found = joint_tree.knnSearch(&query[0], real_k,
                                            &ret_idx[0], &ret_dist[0]);

        // Max distance among k+1 neighbors (includes self at dist 0)
        double max_dist = *std::max_element(ret_dist.begin(),
                                            ret_dist.begin() + found);

        // Nudge down by 1 ULP — matching knnmi's nextafter trick
        eps_vec[i] = std::nextafter(max_dist, 0.0);
    }

    // --- Marginal radius searches ---
    double digamma_sum = 0.0;
    nanoflann::SearchParams search_params(10);

    for (int i = 0; i < n; i++) {
        // X marginal: radiusSearch with eps from joint space
        std::vector<double> query_x(d_x);
        for (int j = 0; j < d_x; j++) query_x[j] = x_mat(i, j);

        std::vector<std::pair<uint32_t, double>> x_matches;
        double n_x = x_tree.radiusSearch(&query_x[0], eps_vec[i],
                                         x_matches, search_params);

        // Y marginal: same eps
        std::vector<double> query_y(d_y);
        for (int j = 0; j < d_y; j++) query_y[j] = y_mat(i, j);

        std::vector<std::pair<uint32_t, double>> y_matches;
        double n_y = y_tree.radiusSearch(&query_y[0], eps_vec[i],
                                         y_matches, search_params);

        // radiusSearch returns count including self (dist 0 <= eps)
        // so digamma(n_x) already has the +1 built in.
        digamma_sum += Rf_digamma(n_x) + Rf_digamma(n_y);
    }

    // MI = psi(k) + psi(N) - <psi(n_x) + psi(n_y)>
    // where n_x, n_y include self (matching knnmi convention)
    double mi = Rf_digamma((double)k) + Rf_digamma((double)n)
                - digamma_sum / (double)n;

    return std::max(0.0, mi);
}


// =============================================================================
// R-facing C wrapper
// =============================================================================

extern "C" {

SEXP _mutual_inf_cc_mv(SEXP r_x, SEXP r_y, SEXP r_dx, SEXP r_dy, SEXP r_k) {
    /*
     * Multivariate MI: MI(X; Y)
     *
     * r_x  — numeric vector of length N*d_x (column-major matrix from R)
     * r_y  — numeric vector of length N*d_y (column-major matrix from R)
     * r_dx — integer, number of X dimensions
     * r_dy — integer, number of Y dimensions
     * r_k  — integer, number of nearest neighbors
     *
     * Returns: numeric vector of length 1 (the MI estimate)
     */
    int d_x = INTEGER(r_dx)[0];
    int d_y = INTEGER(r_dy)[0];
    int k = INTEGER(r_k)[0];
    int n = LENGTH(r_x) / d_x;  // N = total elements / number of columns

    SEXP mi = PROTECT(allocVector(REALSXP, 1));
    REAL(mi)[0] = mutual_inf_cc_mv_impl(
        REAL(r_x), n, d_x,
        REAL(r_y), d_y,
        k
    );
    UNPROTECT(1);
    return mi;
}

} // extern "C"
