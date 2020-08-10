#' Gompertz cdf python function
#' @export
pgompertz <- function(x) {
  1 - tf$math$exp(-tf$math$exp(x))
}

#' Gumbel cdf python function
#' @export
pgumbel <- function(x) {
  tf$math$exp(-tf$math$exp(-x))
}
