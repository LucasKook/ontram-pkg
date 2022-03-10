#' Keras interface to ONTRAMs
#' @examples
#' library(tram)
#' set.seed(2021)
#' mbl <- k_mod_baseline(5L, name = "baseline")
#' msh <- mod_shift(2L, name = "linear_shift")
#' mim <- mod_shift(1L, name = "complex_shift")
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
#' compile(m, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
#' cent <- metric_cqwk(ncol(Y))
#' compile(m, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001),
#' metrics = c(cent))
#' fit(m, x = list(INT, X, Z), y = Y, batch_size = nrow(wine), epoch = 10,
#'     view_metrics = FALSE)
#'
#' idx <- 8
#' loss(Y[idx, , drop = FALSE], m(list(INT[idx, , drop = FALSE],
#'      X[idx, , drop = FALSE], Z[idx, , drop = FALSE])))
#'
#' tm <- Polr(rating ~ temp + contact + noise, data = wine)
#' logLik(tm, newdata = wine[idx,])
#'
#' tmp <- get_weights(m)
#' tmp[[1]][] <- .to_gamma(coef(as.mlt(tm))[1:4])
#' tmp[[2]][] <- coef(tm)[1:2]
#' tmp[[3]][] <- coef(tm)[3]
#' set_weights(m, tmp)
#'
#' loss(k_constant(Y), m(list(INT, X, Z)))
#' - logLik(tm)
#' - logLik(tm) / nrow(wine)
#'
#' @export
k_ontram <- function(
  mod_baseline,
  list_of_shift_models = NULL,
  complex_intercept = FALSE,
  ...
) {
  if (is.null(list_of_shift_models)) {
    list_of_shift_models <- keras_model_sequential() %>%
      layer_dense(units = 1L, input_shape = c(1L),
                  kernel_initializer = initializer_zeros(),
                  use_bias = FALSE, trainable = FALSE)
  }
  nshift <- length(list_of_shift_models)
  if (nshift == 1L) {
    shift_in <- list_of_shift_models$input
    shift_out <- list_of_shift_models$output
  } else if (nshift >= 2L) {
    shift_in <- lapply(list_of_shift_models, function(x) x$input)
    shift_out <- lapply(list_of_shift_models, function(x) x$output) %>%
      layer_add()
  }
  inputs <- list(mod_baseline$input, shift_in)
  outputs <- list(mod_baseline$output, shift_out)
  m <- keras_model(inputs = inputs, outputs = layer_concatenate(outputs))
  m$mod_baseline <- mod_baseline
  m$list_of_shift_models <- list_of_shift_models
  if (complex_intercept) {
    class(m) <- c("k_ontram_ci", "k_ontram", class(m))
  } else {
    class(m) <- c("k_ontram", class(m))
  }
  return(m)
}

#' Layer for transforming raw intercepts using softplus function
#' @export
layer_trafo_intercept <- function() {
  tf$keras$layers$Lambda(
    function(x) {
      w1 <- x[, 1L, drop = FALSE]
      wrest <- tf$math$softplus(x[, 2L:x$shape[[2]], drop = FALSE])
      tf$cumsum(k_concatenate(list(w1, wrest), axis = 0L), axis = 1L)
    }
  )
}

#' Baseline model
#' @export
k_mod_baseline <- function(K, ...) {
  keras_model_sequential() %>%
    layer_dense(units = K - 1L, input_shape = 1L, use_bias = FALSE,
                ... = ...) %>%
    layer_trafo_intercept()()
}

#' gamma to theta
#' @examples
#' .to_theta(c(-1, 1, 1))
.to_theta <- function(gammas) {
  return(c(gammas[1], gammas[1] + cumsum(log(1 + exp(gammas[-1])))))
}

#' theta to gamma
#' @examples
#' .to_gamma(.to_theta(c(-1, 1, 1)))
.to_gamma <- function(thetas) {
  gammas <- c(thetas[1L], log(exp(diff(thetas)) - 1))
  if(any(!is.finite(gammas))) {
    gammas[!is.finite(gammas)] <- 1e-20
  }
  return(gammas)
}
