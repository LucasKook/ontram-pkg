#' Keras interface to ONTRAMs
#' @examples
#' library(tram)
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
#' compile(m, loss = loss, optimizer = optimizer_adam(lr = 1e-2, decay = 0.001))
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
#'
#' @export
k_ontram <- function(
  mod_baseline,
  list_of_shift_models = NULL,
  ...
) {
  if (is.null(list_of_shift_models)) {
    list_of_shift_models <- keras_model_sequential() %>%
      layer_dense(units = 1L, input_shape = c(1L),
                  kernel_initializer = initializer_zeros(),
                  trainable = FALSE)
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
  class(m) <- c("k_ontram", class(m))
  return(m)
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
    yu <- y_true[, 1L:(K - 1L), drop = FALSE]
    yl <- y_true[, 2L:K, drop = FALSE]
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
#' @examples
#' layer_trafo_intercept()
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

#' S3 methods for \code{k_ontram}
#' @method predict k_ontram
#' @export
predict.k_ontram <- function(object, x,
                             type = c("distribution", "density", "trafo",
                                      "baseline_only", "hazard", "cumhazard",
                                      "survivor", "odds"),
                             ...) {
  type <- match.arg(type)
  class(object) <- class(object)[-1L]
  preds <- predict(object, x = x, ... = ...)
  K <- ncol(preds)
  baseline <- preds[, 1L:(K - 1L)]
  shift <- do.call("cbind", lapply(1L:(K - 1L), function(x) preds[, K]))
  trafo <- baseline - shift
  ccdf <- cbind(plogis(trafo), 1)
  cdf <- cbind(0, ccdf)
  pdf <- t(apply(cdf, 1, diff))
  surv <- 1 - ccdf
  haz <- pdf / (1 - ccdf)
  cumhaz <- - log(surv)
  odds <- ccdf / (1 - ccdf)

  ret <- switch(type,
                "distribution" = cdf,
                "density" = pdf,
                "trafo" = trafo,
                "baseline_only" = baseline,
                "hazard" = haz,
                "cumhazard" = cumhaz,
                "survivor" = surv,
                "odds" = odds)

  return(ret)
}

#' Simulate Responses
#' @method simulate k_ontram
#' @param object an object of class \code{\link{k_ontram}}.
#' @param x list of data matrices (including matrix containing 1 if model intercept is non-complex)
#' @param nsim number of simulations.
#' @param levels levels of simulated ordered responses.
#' @param seed random seed.
#' @examples
#' data(wine, package = "ordinal")
#' fm <- rating ~ temp + contact
#' y <- model.matrix(~ 0 + rating, data = wine)
#' x <- ontram:::.rm_int(model.matrix(fm, data = wine))
#' loss <- k_ontram_loss(ncol(y))
#'
#' mbl <- k_mod_baseline(ncol(y), name = "baseline")
#' msh <- mod_shift(ncol(x), name = "linear_shift")
#'
#' mo <- k_ontram(mbl, msh)
#' compile(mo, optimizer = optimizer_adam(learning_rate = 10^-4), loss = loss)
#' fit(mo, x = list(matrix(1, nrow = nrow(wine)), x), y = y, batch_size = nrow(wine), epoch = 10)
#' simulate(mo, x = list(matrix(1, nrow = nrow(wine)), x), nsim = 1)
#' @export
simulate.k_ontram <- function(object, x, nsim = 1, levels = NULL, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  pr <- predict(object, x = x, type = "density")
  if (is.null(levels)) {
    levels <- 1:ncol(pr)
  }
  ret <- apply(pr, 1, function(p) sample(levels, nsim, prob = p, replace = TRUE))
  if (nsim > 1) {
    tmp <- vector(mode = "list", length = nsim)
    for (i in 1:nsim) {
      tmp[[i]] <- ordered(ret[i, ], levels = levels)
    }
    ret <- tmp
  } else {
    ret <- ordered(ret, levels = levels)
  }
  return(ret)
}


warm_start.ontram <- function(model, tram, which = c("all", "baseline only", "shift only")) {
  stopifnot(which %in% c("all", "baseline only", "shift only"))
  K <- model$y_dim
  x_dim <- model$x_dim
  w <- vector(mode = "list", length = 2)
  names(w) <- c("w_baseline", "w_shift")
  w$w_baseline <- list(matrix(coef(tram, with_baseline = T)[1:K-1],
                              nrow = 1, ncol = K-1))
  if (!is.null(coef(tram))) {
    w$w_shift <- list(matrix(coef(tram),
                             nrow = x_dim, ncol = 1))
  }
  if (which %in% "baseline only") {
    model$mod_baseline$set_weights(w$w_baseline)
  }
  if (which %in% "shift only") {
    model$mod_shift$set_weights(w$w_shift)
  }
  if (which %in% "all") {
    model$mod_baseline$set_weights(w$w_baseline)
    model$mod_shift$set_weights(w$w_shift)
  }
  return(invisible(model))
}

warm_start.k_ontram <- function(object, tram, which = c("all", "baseline only", "shift only")) {
  which <- match.arg(which)
  w_old <- get_weights(object)
  w_new <- vector(mode = "list", length = 2)
  w_new$w_baseline <- list(matrix(coef(tram, with_baseline = T)[1:K-1],
                                  nrow = 1, ncol = K-1))
  if (!is.null(coef(tram))) {
    w$w_shift <- list(matrix(coef(tram),
                             nrow = x_dim, ncol = 1))
  }

}
