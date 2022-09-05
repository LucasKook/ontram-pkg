#' Sequential model for the intercept function
#' @description Fully connected neural network with input dimension K and
#'     output dimension K - 1
#' @return Returns a model of type \code{\link[keras]{keras_mod_sequential}}
#' @importFrom keras keras_model_sequential layer_dense
#' @examples
#' m <- mod_baseline(5L)
#' @export
mod_baseline <- function(y_dim, ...) {
  mod <- keras_model_sequential() %>%
    layer_dense(units = y_dim - 1L, input_shape = 1L, use_bias = FALSE, ... = ...)
  return(mod)
}

#' Transform raw intercept function to constrained one
#' @description Converts raw intercept function, which is the output of
#'     \code{\link{mod_baseline}}, to the valid intercept function, which
#'     is used to evaluate the likelihood and it's gradient.
#'     Namely, theta_1 = gamma_1, theta_k = theta_1 + sum_{i = 1}^{k-1}
#'     exp(gamma_i) and by convention theta_0 = -Inf, theta_K = Inf.
#' @return Returns a \code{tf.tensor} containing theta_0 to theta_K
#' @importFrom keras k_reshape k_exp k_constant k_concatenate
#' @importFrom tensorflow tf
#' @examples
#' gammas <- tf$constant(c(0, -1, -1, -1), shape = c(1L, 4L), dtype = "float32")
#' gamma_to_theta(gammas)
#' @export
gamma_to_theta <- function(gammas) {
  grows <- as.integer(nrow(gammas))
  gcols <- as.integer(ncol(gammas))
  # if (gcols == 1L)
    # return(gammas)
  theta1 <- k_reshape(gammas[, 1L], shape = c(grows, 1L))
  thetas <- k_exp(k_reshape(gammas[, 2L:gcols], shape = c(grows, gcols - 1L)))
  theta0 <- k_constant(-.Machine$double.xmax^0.1, shape = c(grows, 1L))
  thetaK <- k_constant(.Machine$double.xmax^0.1, shape = c(grows, 1L))
  ret <- k_concatenate(
    c(theta0, theta1, theta1 + tf$math$cumsum(thetas, axis = 1L), thetaK),
    axis = -1L)
  return(ret)
}

