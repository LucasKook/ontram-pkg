# Internal for checks

#' gamma to theta
#' @examples
#' .to_theta(c(-1, 1, 1))
.to_theta <- function(gammas) {
  return(c(gammas[1], gammas[1] + cumsum(exp(gammas[-1]))))
}

#' theta to gamma
#' @examples
#' .to_gamma(.to_theta(c(-1, 1, 1)))
.to_gamma <- function(thetas) {
  return(c(thetas[1L], log(diff(thetas))))
}

#' @importFrom tensorflow tf
.switch_method <- function(method) {
  ret <- switch(method, "logit" = tf$sigmoid,
                "cloglog" = tf_function(pgompertz),
                "loglog" = tf_function(pgumbel),
                "probit" = stop("Not implemented yet."))
  return(ret)
}
