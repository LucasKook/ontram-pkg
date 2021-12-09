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
  gammas <- c(thetas[1L], log(diff(thetas)))
  if(any(is.nan(gammas))) {
    gammas[is.nan(gammas)] <- 1e-20
  }

}

#' @importFrom tensorflow tf
.switch_method <- function(method) {
  ret <- switch(method, "logit" = tf$sigmoid,
                "cloglog" = tf_function(pgompertz),
                "loglog" = tf_function(pgumbel),
                "probit" = stop("Not implemented yet."))
  return(ret)
}

#' Get weights by name of layer
#' @export
get_weights_by_name <- function(m, name) {
  layer_names <- unlist(lapply(m$layers, function(x) x$name))
  layer_weights <- lapply(m$layers, function(x) x$weights)
  idx <- which(layer_names == name)
  as.vector(layer_weights[[idx]][[1]]$numpy())
}

.rm_int <- function(x) {
  if (all(x[, 1] == 1))
    return(x[, -1L, drop = FALSE])
  return(x)
}
