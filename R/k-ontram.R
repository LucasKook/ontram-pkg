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
#' ## Complex intercept ##
#'
# mim2 <- keras_model_sequential() %>%
#    layer_dense(units = 4, input_shape = 1L, activation = "relu") %>%
#    layer_dense(units = ncol(Y) - 1, use_bias = FALSE)
#
# mbl2 <- k_mod_baseline(ncol(Y), mod_complex = mim2)
# m2 <- k_ontram(mbl2)
# compile(m2, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
# fit(m2, x = list(INT, Z), y = Y, batch_size = nrow(wine), epoch = 10,
#     view_metrics = FALSE)
#
# get_weights(m2$mod_baseline[[0]])
# warmstart(m2, tm, "baseline only")
# get_weights(m2$mod_baseline[[0]])
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
  m$mod_baseline <- mod_baseline
  m$list_of_shift_models <- list_of_shift_models
  if (complex_intercept) {
    class(m) <- c("k_ontram_ci", "k_ontram", class(m))
  } else {
    class(m) <- c("k_ontram", class(m))
  }
  return(m)
}

#' Function for estimating the model
#' @examples
#' set.seed(2021)
#' mbl <- k_mod_baseline(5L, name = "baseline")
#' msh <- mod_shift(2L, name = "linear_shift")
#' mim <- mod_shift(1L, name = "complex_shift")
#' m <- k_ontram(mbl, msh)
#' m2 <- k_ontram(mbl, list(msh, mim))
#'
#' data("wine", package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' X <- ontram:::.rm_int(model.matrix(~ temp + contact, data = wine))
#' Z <- ontram:::.rm_int(model.matrix(~ noise, data = wine))
#' Y <- model.matrix(~ 0 + rating, data = wine)
#'
#' loss <- k_ontram_loss(ncol(Y))
#' compile(m, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
#' fit_k_ontram(m, x = X, y = Y, batch_size = nrow(wine), epoch = 10,
#'              view_metrics = FALSE)
#' compile(m2, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
#' fit_k_ontram(m2, x = list(X, Z), y = Y, batch_size = nrow(wine), epoch = 10,
#'              view_metrics = FALSE)
#' @export
fit_k_ontram <- function(object, x, validation_data = NULL, ...) {
  if (!("k_ontram_ci" %in% class(object))) {
    if (is.list(x)) {
      x <- c(list(matrix(1, nrow = nrow(x[[1]]))), x)
    } else {
      x <- c(list(matrix(1, nrow = nrow(x))), list(x))
    }
    if (!is.null(validation_data)) {
      if (is.list(validation_data[[1]])) {
        validation_data <- list(c(list(matrix(1, nrow = nrow(validation_data[[2]]))),
                                  validation_data[[1]]),
                                validation_data[[2]])
      } else {
        validation_data <- list(c(list(matrix(1, nrow = nrow(validation_data[[2]]))),
                                  list(validation_data[[1]])),
                                validation_data[[2]])
      }
    }
  }
    fit(object, x = x, validation_data = validation_data, ...)
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
layer_trafo_intercept <- function() {
  tf$keras$layers$Lambda(
    function(x) {
      w1 <- x[, 1L, drop = FALSE]
      wrest <- tf$math$exp(x[, 2L:x$shape[[2]], drop = FALSE])
      tf$cumsum(k_concatenate(list(w1, wrest), axis = 0L), axis = 1L)
    }
  )
}

#' keras mbl
#' @examples
#' mbl <- k_mod_baseline(5)
#' mbl(matrix(1))
#' @export
k_mod_baseline <- function(K, ...) {
  keras_model_sequential() %>%
    layer_dense(units = K - 1L, input_shape = 1L, use_bias = FALSE,
                ... = ...) %>%
    layer_trafo_intercept()()
}

#' Reparametrized keras mbl
#' @details \eqn{\nu_k = \nu_k' + \epsilon}
#' @examples
#' library(tram)
#' data("wine", package = "ordinal")
#' mod_polr <- Polr(rating ~ 0, data = wine)
#' thetas <- coef(mod_polr, with_baseline = TRUE)
#' m <- k_mod_baseline_reparametrized(5L, thetas)
#' get_weights(m)
#' @export
k_mod_baseline_reparametrized <- function(K, thetas, ...) {
  bias_fixed <- function(x) {
    k_constant(gammas)
  }
  gammas <- k_constant(.to_gamma(thetas))
  keras_model_sequential() %>%
    layer_dense(units = K - 1L, input_shape = 1L, use_bias = TRUE,
                bias_initializer = initializer_constant(gammas),
                bias_constraint = bias_fixed,
                ... = ...) %>%
    layer_trafo_intercept()()
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
  if ("k_ontram_ci" %in% class(object)) {
    class(object) <- class(object)[-2L]
  } else {
    class(object) <- class(object)[-1L]
    if (is.list(x)) {
      x <- c(list(matrix(1, nrow = nrow(x[[1]]))), x)
    } else {
      x <- c(list(matrix(1, nrow = nrow(x))), list(x))
    }
  }
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

#' Set initial weights
#' @method warmstart k_ontram
#' @param object an object of class \code{\link{k_ontram}}.
#' @param thetas intercepts of a \code{\link[tram]{Polr}} model as a vector (used to initialize the weights of the baseline model).
#' @param betas shift terms of a \code{\link[tram]{Polr}} model as a vector (used to initialize the weights of the simple shift model).
#' @examples
#' library(tram)
#' set.seed(2021)
#' data(wine, package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' y <- model.matrix(~ 0 + rating, data = wine)
#' x <- ontram:::.rm_int(model.matrix(rating ~ temp + contact, data = wine))
#' im <- ontram:::.rm_int(model.matrix(rating ~ noise, data = wine))
#' loss <- k_ontram_loss(ncol(y))
#' mod_polr <- Polr(rating ~ temp + contact, data = wine)
#' mbl <- k_mod_baseline(ncol(y), name = "baseline")
#' msh <- mod_shift(ncol(x), name = "shift")
#' mim <- keras_model_sequential() %>%
#'   layer_dense(units = 8, input_shape = 1L, activation = "relu") %>%
#'   layer_dense(units = 16, activation = "relu") %>%
#'   layer_dense(units = 1, use_bias = FALSE)
#' mo <- k_ontram(mbl, list(msh, mim))
#' mo <- warmstart(mo, thetas = coef(mod_polr, with_baseline = T)[1:4L],
#'                 which = "baseline only")
#' @export
warmstart.k_ontram <- function(object, thetas = NULL, betas = NULL,
                               which = c("all", "baseline only", "shift only")) {
  which <- match.arg(which)
  K <- object$mod_baseline$output_shape[[2L]] + 1
  nshift <- length(object$list_of_shift_models)
  if (which == "all" || which == "baseline only") {
    gammas <- list(matrix(.to_gamma(thetas), nrow = 1, ncol = K-1))
    set_weights(object$mod_baseline, gammas)
  }
  if (which == "all" || which == "shift only") {
  betas <- list(matrix(betas, nrow = length(betas), ncol = 1))
    if (nshift == 1L) {
      set_weights(object$list_of_shift_models, betas)
    } else if (nshift >= 2L) {
      for (idx in 0:nshift - 1L) {
        if (nrow(get_weights(object$list_of_shift_models[[idx]])[[1]]) == nrow(betas[[1]])) {
          set_weights(object$list_of_shift_models[[idx]], betas)
          break
        }
      }
    }
  }
  return(invisible(object))
}

#' Set initial weights
#' @param thetas intercepts of a \code{\link[tram]{Polr}} model as a vector (are added to the last layer during training).
#' @param betas shift terms of a \code{\link[tram]{Polr}} model as a vector (used to initialize the weights of the simple shift model).
#' @section IMPORTANT:
#' \itemize{
#' \item{The warmstarted model has to be assigned to a new variable.}
#' }
#' @examples
#' library(tram)
#' set.seed(2021)
#' data(wine, package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' y <- model.matrix(~ 0 + rating, data = wine)
#' x <- ontram:::.rm_int(model.matrix(rating ~ temp + contact, data = wine))
#' im <- ontram:::.rm_int(model.matrix(rating ~ noise, data = wine))
#' loss <- k_ontram_loss(ncol(y))
#' mbl <- keras_model_sequential() %>%
#'   layer_dense(units = 8, input_shape = 1L, activation = "relu") %>%
#'   layer_dense(units = 16, activation = "relu") %>%
#'   layer_dense(units = ncol(y) - 1, use_bias = FALSE) %>%
#'   layer_trafo_intercept()()
#' mo <- k_ontram(mbl, msh, complex_intercept = TRUE)
#' mo <- warmstart(mo, thetas = coef(mod_polr, with_baseline = TRUE)[1:4L],
#'                 betas = coef(mod_polr), which = "all")
#' @export
warmstart.k_ontram_ci <- function(object, thetas = NULL, betas = NULL,
                                  which = c("all", "baseline only", "shift only")) {
  which <- match.arg(which)
  K <- object$output_shape[[2L]]
  nshift <- length(object$list_of_shift_models)
  if (which == "all" || which == "baseline only") {
    gammas <- .to_gamma(thetas)
    mbl_copy <- clone_model(object$mod_baseline)
    pop_layer(mbl_copy) # remove lambda layer
    ll <- length(mbl_copy$layers)
    ll_activation <- mbl_copy$layers[[ll]]$activation
    mbl_new <- mbl_copy %>%
      layer_add_gamma_tilde(gammas)() %>% # add gammas
      layer_trafo_intercept()()
    object <- k_ontram(mbl_new, object$list_of_shift_models,
                       complex_intercept = TRUE)
  }
  if (which == "all" || which == "shift only") {
   betas <- list(matrix(betas, nrow = length(betas), ncol = 1))
    if (nshift == 1L) {
      set_weights(object$list_of_shift_models, betas)
    } else if (nshift >= 2L) {
      for (idx in 0:nshift - 1L) {
        if (nrow(get_weights(object$list_of_shift_models[[idx]])[[1]]) == nrow(betas[[1]])) {
          set_weights(object$list_of_shift_models[[idx]], betas)
          break
        }
      }
    }
  }
  return(invisible(object))
}

layer_add_gamma_tilde <- function(gammas) {
  gammas <- k_constant(gammas)
  tf$keras$layers$Lambda(
    function(x) {
      tf$math$add(x, gammas)
    }
  )
}

