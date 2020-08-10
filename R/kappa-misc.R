#' Weighting scheme for Cohen's kappa
#' @export
weight_scheme <- function(K, p) {
  outer(1:K, 1:K, function(x, y) abs(x - y)^p / (K - 1)^p)
}

#' Weighted kappa loss function (R)
#' @export
kappa_loss <- function(target, predicted, weights) {
  idx <- apply(target, 1, which.max)
  wgts <- weights[idx, , drop = FALSE]
  num <- sum(wgts * predicted)

  fi <- colSums(target) / sum(target)
  den <- sum(fi * weights %*% colSums(predicted))

  ret <- log(num) - log(den)
  return(ret)
}

#' Expected weighted kappa score
#' @export
expected_score <- function(target, predicted, p = 2) {
  score <- kappa_loss(target, predicted, weight_scheme(ncol(target), p))
  ret <- sum(score * target)
  return(ret)
}

#' expected number of agreements under independence
#' @export
get_expected <- function(confmat) {
  ret <- confmat
  for (i in 1:nrow(confmat)) {
    for (j in 1:ncol(confmat)) {
      ret[i, j] = sum(confmat[i, ]) * sum(confmat[, j]) / sum(confmat)
    }
  }
  return(ret)
}

#' compute Cohen's weighted kappa
#' @export
compute_kappa <- function(confmat, weights) {
  expected <- get_expected(confmat)
  po <- sum(confmat * weights / sum(confmat))
  pe <- sum(expected * weights / sum(expected))
  # ret <- (po - pe)/(1 - pe) # Different parameterization, requires 1 - weights
  ret <- 1 - po / pe
  return(ret)
}

#' get kappa loss python function
#' @export
get_loss <- function(weights) {
  weights <- k_constant(weights)
  loss <- function(y_true, y_pred) {
    idx <- k_argmax(y_true)
    wgts <- tf$gather(weights, idx, axis = 0L)
    num <- k_sum(wgts * y_pred)
    fi <- tf$cast(k_sum(y_true, axis = 1L) / k_sum(y_true), dtype = "float32")
    den <- k_sum(fi * tf$linalg$matvec(weights, k_sum(y_pred, axis = 1L)))
    ret <- k_log(num + 1e-32) - k_log(den + 1e-32)
    return(ret)
  }
  return(loss)
}

#' get kappa metric python function
#' @export
get_metric <- function(K, p) {
  weights <- weight_scheme(K, p)
  qwk_metric <- function(y_true, y_pred) {
    1 - exp(get_loss(weights)(y_true, y_pred))
  }
  qwk <- custom_metric("qwk", qwk_metric)
  return(qwk)
}
