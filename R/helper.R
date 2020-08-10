# Internal for checks
.to_theta <- function(gammas) {
  return(c(gammas[1], gammas[1] + cumsum(exp(gammas[-1]))))
}

#' @importFrom tensorflow tf
.switch_method <- function(method) {
  ret <- switch(method, "logit" = tf$sigmoid,
                "cloglog" = tf_function(pgompertz),
                "loglog" = tf_function(pgumbel),
                "probit" = stop("Not implemented yet."))
  return(ret)
}
