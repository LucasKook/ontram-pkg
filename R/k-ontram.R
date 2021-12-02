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
#' cent <- metric_ontram_crossent(ncol(Y))
#' compile(m, loss = loss, optimizer = optimizer_adam(lr = 1e-2, decay = 0.001),
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
#' - logLik(tm) / nrow(wine)
#'
#' @export
k_ontram <- function(
  mod_baseline,
  list_of_shift_models,
  ...
) {
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
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = TRUE]
    yu <- y_true[, 1L:(K - 1L), drop = FALSE]
    yl <- y_true[, 2L:K, drop = FALSE]
    upr <- k_sum(tf$multiply(yu, intercepts), axis = 0L) - shifts
    lwr <- k_sum(tf$multiply(yl, intercepts), axis = 0L) - shifts
    t1 <- y_true[, 1L, drop = TRUE]
    tK <- y_true[, K, drop = TRUE]
    lik <- t1 * k_sigmoid(upr) + tK * (1 - k_sigmoid(lwr)) +
      (1 - t1) * (1 - tK) * (k_sigmoid(upr) - k_sigmoid(lwr))
    - k_mean(k_log(lik))
  }
}

#' NLL metric
#' @export
metric_nll <- function(K) {
  custom_metric("k_nll", k_ontram_loss(K))
}

#' CRPS loss
#' @examples
#' rps_loss <- k_ontram_rps(ncol(Y))
#' @export
k_ontram_rps <- function(K) {
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = FALSE]
    y_cum <- tf$cumsum(y_true, axis = 1L)
    cdf <- k_sigmoid(intercepts - shifts)
    briers <- (cdf - y_cum[, 1L:(K - 1L), drop = FALSE])^2
    k_mean(k_mean(briers, axis = 1L))
  }
}

#' CRPS metric
#' @examples
#' k_rps <- metric_rps(ncol(Y))
#' k_rps(k_constant(Y), m$output)
#' k_rps(k_constant(Y), m(list(INT, X, Z)))
#' @export
metric_rps <- function(K) {
  custom_metric("k_rps", k_ontram_rps(K))
}

#' Binary logLik function
#' @examples
#' k_binll <- k_ontram_binll(ncol(Y))
#' k_binll(k_constant(Y), m(list(INT, X, Z)))
#' k_binll(k_constant(Y), m$output)
k_ontram_binll <- function(K, cutoff = 3L) {
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = FALSE]
    cdf <- k_sigmoid(intercepts - shifts)
    pbin <- cdf[, cutoff, drop = TRUE]
    ybin <- k_sum(y_true[, 1L:cutoff, drop = FALSE], axis = 0L)
    k_mean(k_binary_crossentropy(ybin, pbin))
  }
}

#' Binary NLL metric
#' @export
metric_binll <- function(K, cutoff = 3L) {
  custom_metric("k_binll", k_ontram_binll(K, cutoff))
}

#' AUC function
#' @examples
#' k_auc <- k_ontram_auc(ncol(Y))
#' k_auc(k_constant(Y), m(list(INT, X, Z)))
#' k_auc(k_constant(Y), m$output)
k_ontram_auc <- function(K, cutoff = 3L) {
  k_AUC <- tf$keras$metrics$AUC()
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = FALSE]
    cdf <- k_sigmoid(intercepts - shifts)
    pbin <- cdf[, cutoff, drop = TRUE]
    ybin <- k_sum(y_true[, 1L:cutoff, drop = FALSE], axis = 0L)
    k_AUC(ybin, pbin)
  }
}

#' Accuracy metric
#' @export
metric_k_auc <- function(K, cutoff = 3L) {
  custom_metric("k_auc", k_ontram_auc(K, cutoff))
}

#' Accuracy function
#' @examples
#' k_acc <- k_ontram_acc(ncol(Y))
#' k_acc(k_constant(Y), m(list(INT, X, Z)))
#' k_acc(k_constant(Y), m$output)
k_ontram_acc <- function(K) {
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = FALSE]
    cdf <- k_sigmoid(intercepts - shifts)
    p1 <- cdf[, 1L, drop = FALSE]
    pK <- 1 - cdf[, K - 1L, drop = FALSE]
    pmf <- k_concatenate(list(p1, cdf[, 2L:(K - 1L), drop = FALSE] -
                                cdf[, 1L:(K - 2L), drop = FALSE], pK))
    k_mean(tf$metrics$categorical_accuracy(y_true, pmf))
  }
}

#' Accuracy metric
#' @export
metric_acc <- function(K) {
  custom_metric("k_acc", k_ontram_acc(K))
}

#' Continuous qwk
#' @examples
#' k_qwk <- k_ontram_qwk(ncol(Y))
#' k_qwk(k_constant(Y), m(list(INT, X, Z)))
#' k_qwk(k_constant(Y), m$output)
#' @export
k_ontram_qwk <- function(K, p = 2L) {
  weights <- k_constant(weight_scheme(K, p))
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = FALSE]
    cdf <- k_sigmoid(intercepts - shifts)
    p1 <- cdf[, 1L, drop = FALSE]
    pK <- 1 - cdf[, K - 1L, drop = FALSE]
    pmf <- k_concatenate(list(p1, cdf[, 2L:(K - 1L), drop = FALSE] -
                                cdf[, 1L:(K - 2L), drop = FALSE], pK))
    idx <- k_argmax(y_true)
    wgts <- tf$gather(weights, idx, axis = 0L)
    num <- k_sum(wgts * pmf)
    fi <- k_sum(y_true, axis = 1L) / k_sum(y_true)
    den <- k_sum(fi * tf$linalg$matvec(weights, k_sum(pmf, axis = 1L)))
    k_log(num + 1e-32) - k_log(den + 1e-32)
  }
}

#' Continuous qwk metric
#' @export
metric_qwk <- function(K, p = 2L) {
  met <- function(y_true, y_pred) 1 - k_exp(k_ontram_qwk(K, p)(y_true, y_pred))
  custom_metric("k_qwk", met)
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
  mbl <- keras_model_sequential() %>%
    layer_dense(units = K - 1L, input_shape = 1L, use_bias = FALSE,
                ... = ...)
  to_theta <- layer_trafo_intercept()
  keras_model(mbl$input, to_theta(mbl$output))
}

#' S3 methods for \code{k_ontram}
#' @method predict k_ontram
#' @export
predict.k_ontram <- function(object, x,
                             type = c("distribution", "density", "trafo",
                                      "baseline_only", "hazard", "cumhazard",
                                      "survivor", "odds", "raw"),
                             ...) {
  type <- match.arg(type)
  class(object) <- class(object)[-1L]
  preds <- predict(object, x = x, ... = ...)
  if (type == "raw")
    return(preds)
  K <- ncol(preds)
  baseline <- preds[, 1L:(K - 1L)]
  shift <- do.call("cbind", lapply(1L:(K - 1L), function(x) preds[, K]))
  trafo <- baseline - shift
  ccdf <- cbind(plogis(trafo), 1)
  cdf <- cbind(0, ccdf)
  pdf <- t(apply(cdf, 1, diff))
  surv <- 1 - ccdf
  haz <- pdf / (1 - ccdf)
  cumhaz <-  - log(surv)
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
