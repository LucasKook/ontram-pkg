#' @export
gtt <- function(gammas) {
  grows <- as.integer(nrow(gammas)[[1]])
  gcols <- as.integer(ncol(gammas)[[1]])
  theta1 <- k_reshape(gammas[, 1L], shape = c(grows, 1L))
  thetas <- k_exp(k_reshape(gammas[, 2L:gcols], shape = c(grows, gcols - 1L)))
  theta0 <- k_constant(-.Machine$double.xmax^0.1, shape = c(grows, 1L))
  thetaK <- k_constant(.Machine$double.xmax^0.1, shape = c(grows, 1L))
  ret <- k_concatenate(
    c(theta0, theta1, theta1 + tf$math$cumsum(thetas, axis = 1L), thetaK),
    axis = -1L)
  return(ret)
}

#' @examples
#' mbl <- mod_baseline(7L, name = "baseline")
#' msh <- mod_shift(2L, name = "shift")
#' mim <- mod_shift(2L, name = "img")
#' m <- keras_model(input = list(mbl$input, msh$input, mim$input),
#'     output = layer_concatenate(list(mbl$output, msh$output, mim$output)))
#' y <- matrix(c(0, 1, 0, 0, 0, 0, 0), ncol = 7, nrow = 10, byrow = TRUE)
#' inp <- list(matrix(1, nrow = 10), matrix(1:2, ncol = 2, nrow = 10), matrix(1:2, ncol = 2, nrow = 10))
#' m(inp)
#' loss <- ontram_logLik(7L)
#' compile(m, loss = loss, optimizer = "adam")
#' loss(k_constant(y), m$output)
#' loss(k_constant(y), m(inp))
#' fit(m, x = inp, y = y, batch_size = 10, epochs = 10)
#' @export
ontram_logLik <- function(y_dim) {
  ret <- function(y_true, y_pred) {
    yu <- tf$pad(y_true, matrix(c(0L, 1L, 0L, 0L), ncol = 2))
    yl <- tf$pad(y_true, matrix(c(0L, 0L, 0L, 1L), ncol = 2))
    fwd_theta <- gtt(y_pred[, 1:(y_dim - 1), drop = FALSE])
    fwd_eta <- y_pred[, y_dim, drop = TRUE]
    fwd_beta <- y_pred[, y_dim + 1L, drop = TRUE]
    if (length(dim(fwd_theta)) != 1L) {
      itr <- tf$linalg$diag_part(tf$linalg$matvec(yu, fwd_theta))
      itl <- tf$linalg$diag_part(tf$linalg$matvec(yl, fwd_theta))
    } else {
      itr <- tf$linalg$matvec(yu, fwd_theta)
      itl <- tf$linalg$matvec(yl, fwd_theta)
    }

    lik <- k_sigmoid(itr - fwd_eta - fwd_beta) -
      k_sigmoid(itl - fwd_eta - fwd_beta)
    nll <- - k_mean(k_log(k_clip(lik, 1e-12, 1 - 1e-12)))
    return(nll)
  }
  return(ret)
}

#' @examples
#' mbl <- mod_baseline(7L, name = "baseline")
#' msh <- mod_shift(2L, name = "shift")
#' mim <- mod_shift(2L, name = "img")
#' m <- keras_model(input = list(mbl$input, msh$input, mim$input),
#'     output = layer_concatenate(list(mbl$output, msh$output, mim$output)))
#' y <- matrix(c(0, 1, 0, 0, 0, 0, 0), ncol = 7, nrow = 10, byrow = TRUE)
#' inp <- list(matrix(1, nrow = 10), matrix(1:2, ncol = 2, nrow = 10), matrix(1:2, ncol = 2, nrow = 10))
#' m(inp)
#' loss <- softmax_like(7L)
#' compile(m, loss = loss, optimizer = "adam")
#' loss(k_constant(y), m$output)
#' loss(k_constant(y), m(inp))
#' fit(m, x = inp, y = y, batch_size = 10, epochs = 10)
#' @export
softmax_like <- function(y_dim) {
  ret <- function(y_true, y_pred) {
    yu <- tf$pad(y_true, matrix(c(0L, 1L, 0L, 0L), ncol = 2))
    fwd_theta <- gtt(y_pred[, 1:(y_dim - 1), drop = FALSE])
    fwd_eta <- y_pred[, y_dim, drop = FALSE]
    fwd_beta <- y_pred[, y_dim + 1L, drop = FALSE]
    probs <- k_sigmoid(fwd_theta - fwd_eta - fwd_beta)
    pmf <- probs[, 2:ncol(probs)[[1]]] - probs[, 1:(ncol(probs)[[1]] - 1)]
    nll <- k_mean(k_categorical_crossentropy(y_true, pmf))
    return(nll)
  }
  return(ret)
}
