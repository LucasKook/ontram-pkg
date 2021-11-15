#' Keras interface to ONTRAMs
#' @examples
#' library(tram)
#' mbl <- k_mod_baseline(5L)
#' msh <- mod_shift(2L)
#' mim <- mod_shift(1L)
#' m <- k_ontram(mbl, list(msh, mim))
#'
#' data("wine", package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' X <- .rm_int(model.matrix(~ temp + contact, data = wine))
#' Y <- model.matrix(~ 0 + rating, data = wine)
#' Z <- .rm_int(model.matrix(~ noise, data = wine))
#' INT <- matrix(1, nrow = nrow(wine))
#'
#' m(list(INT, X, Z))
#' loss <- k_ontram_loss(ncol(Y))
#' compile(m, loss = loss, optimizer = optimizer_adam(lr = 1e-3))
#' fit(m, x = list(INT, X, Z), y = Y, batch_size = nrow(wine), epoch = 100L)
#'
#' idx <- 8
#' loss(Y[idx, , drop = FALSE], m(list(INT[idx, , drop = FALSE],
#'      X[idx, , drop = FALSE], Z[idx, , drop = FALSE])))
#' logLik(tm, newdata = wine[idx,])
#'
#' tm <- Polr(rating ~ temp + contact + noise, data = wine)
#'
#' tmp <- get_weights(m)
#' tmp[[1]][] <- .to_gamma(coef(as.mlt(tm))[1:4])
#' tmp[[2]][] <- coef(tm)[1:2]
#' tmp[[3]][] <- coef(tm)[3]
#' set_weights(m, tmp)
#'
#' loss(k_constant(Y), m(list(INT, X, Z)))
#' - logLik(tm)
#'
#' @export
k_ontram <- function(
  mod_baseline,
  list_of_shift_models,
  ...
) {
  inputs <- list(mod_baseline$input,
                 lapply(list_of_shift_models, function(x) x$input))
  outputs <- list(mod_baseline$output,
                  lapply(list_of_shift_models, function(x) x$output) %>%
                    layer_add())
  keras_model(inputs = inputs, outputs = layer_concatenate(outputs))
}

#' Another keras implementation of the ontram loss
#' @examples
#' y_true <- k_constant(matrix(c(1, 0, 0, 0, 0), nrow = 1L))
#' loss <- k_ontram_loss(ncol(y_true))
#' loss(y_true, m$output)
#' debugonce(loss)
#' loss(k_constant(Y), m(list(INT, X, Z)))
#' @export
k_ontram_loss <- function(K) {
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = TRUE]
    shifts <- y_pred[, K, drop = TRUE]
    yu <- deepregression::tf_stride_cols(y_true, 1L, K - 1L)
    yl <- deepregression::tf_stride_cols(y_true, 2L, K)
    upr <- k_sum(tf$multiply(yu, intercepts), axis = 0L) - shifts
    lwr <- k_sum(tf$multiply(yl, intercepts), axis = 0L) - shifts
    t1 <- y_true[, 1L, drop = TRUE]
    tK <- y_true[, K, drop = TRUE]
    lik <- t1 * k_sigmoid(upr) + tK * (1 - k_sigmoid(lwr)) +
      (1 - t1) * (1 - tK) * (k_sigmoid(upr) - k_sigmoid(lwr))
    - k_sum(k_log(lik))
  }
}

#' Layer for transforming raw intercepts
#' @export
layer_trafo_intercept <- tf$keras$layers$Lambda(
  function(x) {
    w1 <- x[, 1L, drop = FALSE]
    wrest <- tf$math$exp(x[, 2L:x$shape[[2]], drop = FALSE])
    tf$cumsum(k_concatenate(list(w1, wrest), axis = 0L), axis = 1L)
  }
)

#' keras mbl
#' @examples
#' mbl <- k_mod_baseline(5)
#' mbl(matrix(1))
#' @export
k_mod_baseline <- function(K, ...) {
  keras_model_sequential() %>%
    layer_dense(units = K - 1L, input_shape = 1L, use_bias = FALSE,
                ... = ...) %>%
    layer_trafo_intercept()
}