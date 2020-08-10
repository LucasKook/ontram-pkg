#' RPS with weights
#' @export
rps <- function(target, predicted, weights = NULL) {
  K <- ncol(target)
  pred <- t(apply(predicted, 1, cumsum))
  targ <- t(apply(target, 1, cumsum))
  urps <- apply((targ - pred)^2, 1, sum)
  if (!is.null(weights))
    urps <- weights * urps
  rps <- mean(urps)/(K-1)
  return(rps)
}
