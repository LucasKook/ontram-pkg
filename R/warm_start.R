#' Set initial weights
#' @export
warm_start <- function(model, model_w, which = c("all", "baseline only", "shift only")) {
  K <- model$y_dim
  x_dim <- model$x_dim
  w <- vector(mode = "list", length = 2)
  names(w) <- c("w_baseline", "w_shift")
  if ("tram" %in% class(model_w)) {
    w$w_baseline <- list(matrix(coef(model_w, with_baseline = T)[1:K-1],
                                nrow = 1, ncol = K-1))
    w$w_shift <- list(matrix(coef(model_w),
                             nrow = x_dim, ncol = 1))
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
  }
  if ("ontram" %in% class(model_w)) {
    w$w_baseline <- model_w$mod_baseline$get_weights()
    if (!is.null(model_w$mod_shift)) {
      w$w_shift <- model_w$mod_shift$get_weights()
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
  }
  return(invisible(model))
}
