
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

#' RPS loss
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
#' k_qwk <- k_ontram_cqwk(ncol(Y))
#' k_qwk(k_constant(Y), m(list(INT, X, Z)))
#' k_qwk(k_constant(Y), m$output)
#' @export
k_ontram_cqwk <- function(K, p = 2L) {
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
metric_cqwk <- function(K, p = 2L) {
  met <- function(y_true, y_pred) 1 - k_exp(k_ontram_cqwk(K, p)(y_true, y_pred))
  custom_metric("k_cqwk", met)
}

#' Discrete qwk
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
    pt <- k_argmax(pmf, axis = 0L)
    yt <- k_argmax(y_true)
    cmat <- tf$cast(tf$math$confusion_matrix(yt, pt), dtype = "float32")
    observed_margin <- k_sum(cmat, axis = 0L)
    predicted_margin <- k_sum(cmat, axis = 1L)

    expected_cmat <- tf$linalg$matmul(
      k_reshape(observed_margin, shape = c(observed_margin$shape[[1L]], 1L)),
      k_reshape(predicted_margin, shape = c(1L, predicted_margin$shape[[1L]]))
    ) / y_true$shape[[1L]]

    # (k_sum(weights * cmat) - k_sum(weights * expected_cmat)) /
    #   (1 - k_sum(weights * expected_cmat))

    1 - k_sum(weights * cmat) / k_sum(weights * expected_cmat)
  }
}

#' Discrete qwk metric
#' @export
metric_qwk <- function(K, p = 2L) {
  custom_metric("k_qwk", k_ontram_qwk(K, p))
}
