#' Compute logLik contributions for exact response
#' @examples
#' gammas <- matrix(c(-1, -1, -1, -1, -2, -1, -1, -1), nrow = 2, byrow = TRUE)
#' gammas <- tf$Variable(gammas, dtype = "float32")
#' beta <- tf$constant(c(1, -1), dtype = "float32")
#' eta <- tf$constant(c(-1, -2), dtype = "float32")
#' y_train <- matrix(c(0, 0, 0, 1, 0, 1, 0, 0, 0, 0), nrow = 2, byrow = TRUE)
#' y_train <- tf$constant(y_train, dtype = "float32")
#' compute_logLik(gammas = gammas, beta = beta, eta = eta, y_train = y_train)
#'
#' gammas <- matrix(c(-1, -1, -1, -1), nrow = 1, byrow = TRUE)
#' gammas <- tf$Variable(gammas, dtype = "float32")
#' beta <- tf$constant(c(1, -1), dtype = "float32")
#' eta <- tf$constant(c(-1, -2), dtype = "float32")
#' y_train <- matrix(c(0, 0, 0, 1, 0, 1, 0, 0, 0, 0), nrow = 2, byrow = TRUE)
#' y_train <- tf$constant(y_train, dtype = "float32")
#' compute_logLik(gammas = gammas, beta = beta, eta = eta, y_train = y_train)
#' @export
compute_logLik <- function(gammas, beta = NULL, eta = NULL, y_train,
                           distr = k_sigmoid) {
  thetas <- gamma_to_theta(gammas)
  yu <- tf$pad(y_train, matrix(c(0L, 1L, 0L, 0L), ncol = 2))
  yl <- tf$pad(y_train, matrix(c(0L, 0L, 0L, 1L), ncol = 2))
  intr_upper <- tf$linalg$matmul(yu, tf$transpose(thetas))
  intr_lower <- tf$linalg$matmul(yl, tf$transpose(thetas))
  if (all(dim(intr_upper) == c(nrow(yu), nrow(yl)))) {
    intr_upper <- tf$linalg$diag_part(intr_upper)
    intr_lower <- tf$linalg$diag_part(intr_lower)
  }
  if (is.null(beta))
    beta <- k_zeros_like(intr_upper)
  if (is.null(eta))
    eta <- k_zeros_like(intr_upper)
  if (length(dim(beta)) == 1L || length(dim(eta)) == 1L) {
    if (length(dim(intr_upper)) > 1L) {
      intr_upper <- k_squeeze(intr_upper, 2L)
      intr_lower <- k_squeeze(intr_lower, 2L)
    }
  }
  lli <- distr(intr_upper - beta - eta) - distr(intr_lower - beta - eta)
  nll <- - tf$reduce_mean(tf$math$log(lli + 1e-16))
  return(nll)
}
