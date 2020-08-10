#' Combine intercept and tabular model
#' @description Simple proportional odds logistic regression neural network,
#'     without response-varying effects and a simple linear predictor for the
#'     explanatory covariates.
#' @importFrom tensorflow tf
#' @examples
#' ontram_polr(2L, 5L)
#' @export
ontram_polr <- function(x_dim, y_dim,
                        method = c("logit", "cloglog", "loglog", "probit"),
                        n_batches = 16, epochs = 50, lr = 0.001,
                        optimizer = tf$keras$optimizer$Adam(learning_rate = lr)) {
  stopifnot(x_dim > 0L && y_dim > 0L)
  method <- match.arg(method)
  distr <- .switch_method(method)
  mt <- mod_baseline(y_dim)
  mc <- mod_shift(x_dim)
  ret <- list(
    x_dim = x_dim,
    y_dim = y_dim,
    n_batches = n_batches,
    epochs = epochs,
    distr = distr,
    mod_baseline = mt,
    mod_shift = mc,
    optimizer = tf$keras$optimizers$Adam(learning_rate = 0.001),
    method = method,
    response_varying = FALSE
  )
  class(ret) <- c("ontram_polr", "ontram")
  return(ret)
}

#' General ordinal transformation neural network
#' @export
ontram <- function(mod_bl, mod_sh, mod_im = NULL, y_dim, x_dim, img_dim = NULL,
                   method = c("logit", "cloglog", "loglog", "probit"),
                   n_batches = 16, epochs = 50, lr = 0.001,
                   optimizer = tf$keras$optimizers$Adam(learning_rate = lr),
                   response_varying = FALSE) {
  method <- match.arg(method)
  distr <- .switch_method(method)
  ret <- list(
    y_dim = y_dim,
    x_dim = x_dim,
    img_dim,
    n_batches = n_batches,
    epochs = epochs,
    distr = distr,
    mod_baseline = mod_bl,
    mod_shift = mod_sh,
    mod_image = mod_im,
    optimizer = optimizer,
    response_varying = response_varying
  )
  class(ret) <- "ontram"
  if (response_varying)
    class(ret) <- c("ontram_rv", class(ret))
  return(ret)
}

#' initializer for equal class probs
.initializer_bias_gamma <- function(K = 10) {
  thetas <- qlogis(seq(0, 1, length.out = K + 1))
  thetas <- thetas[is.finite(thetas)]
  gamma1 <- thetas[1]
  gammas <- log(diff(thetas))
  return(c(gamma1, gammas))
}
