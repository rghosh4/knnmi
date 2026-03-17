## =============================================================================
## mutual_inf_cc_mv.R
##
## R wrapper for the multivariate MI function. Add this to knnmi/R/knnmi.R
## (or as a separate file in knnmi/R/).
## =============================================================================

#' Multivariate Mutual Information Estimation
#'
#' Estimate MI(X;Y) where X and Y can be multivariate (matrices).
#' Uses the same KSG I^(1) algorithm as \code{mutual_inf_cc}, extended
#' to arbitrary dimensions. The joint space uses Chebyshev (L-infinity)
#' distance, and the marginal neighbor counts use the same coupled-distance
#' trick for internal bias cancellation.
#'
#' @param x Input vector of length N, or matrix of size N x d_x.
#'          If a matrix, columns are dimensions and rows are observations.
#' @param y Input vector of length N, or matrix of size N x d_y.
#'          If a matrix, columns are dimensions and rows are observations.
#' @param k Integer number of nearest neighbors. Default is 5.
#' @return Scalar MI estimate.
#'
#' @details
#' This function computes MI(X; Y) where X is treated as a single
#' d_x-dimensional random variable and Y as a single d_y-dimensional
#' random variable. This is different from \code{mutual_inf_cc} with a
#' matrix argument, which loops over rows computing MI(x; y_i) separately.
#'
#' The algorithm is:
#' \enumerate{
#'   \item Form the joint space Z = (X, Y), dimension d_x + d_y
#'   \item Find the k-th nearest neighbor distance eps_i for each point in Z
#'   \item Count n_x = neighbors of point i in X-space within eps_i
#'   \item Count n_y = neighbors of point i in Y-space within eps_i
#'   \item MI = psi(k) + psi(N) - mean(psi(n_x) + psi(n_y))
#' }
#'
#' Counts include the point itself (self-distance = 0 <= eps), matching
#' the convention of \code{mutual_inf_cc}.
#'
#' @examples
#' set.seed(123)
#' # MI between a scalar X and a 2D vector Y = (Y1, Y2)
#' x <- rnorm(1000)
#' y1 <- x + rnorm(1000)
#' y2 <- 0.5 * x + rnorm(1000)
#' mutual_inf_cc_mv(x, cbind(y1, y2), k = 5)
#'
#' # Bivariate case: should match mutual_inf_cc
#' mutual_inf_cc_mv(x, y1, k = 5)
#' mutual_inf_cc(x, y1, k = 5)
#'
#' @export
mutual_inf_cc_mv <- function(x, y, k = 5L) {
  # Coerce to matrices
  if (is.vector(x)) {
    x <- matrix(x, ncol = 1)
  } else {
    x <- as.matrix(x)
  }
  if (is.vector(y)) {
    y <- matrix(y, ncol = 1)
  } else {
    y <- as.matrix(y)
  }

  n <- nrow(x)
  stopifnot(nrow(y) == n)
  stopifnot(k >= 1L, k < n)

  d_x <- ncol(x)
  d_y <- ncol(y)

  # Call the C function
  .Call("_mutual_inf_cc_mv",
        as.double(x),
        as.double(y),
        as.integer(d_x),
        as.integer(d_y),
        as.integer(k))
}
