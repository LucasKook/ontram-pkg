#' @export
weights.ontram <- function(model) {
  ret <- vector(mode = "list")
  ret$w_baseline <- model$mod_baseline$get_weights()
  if (!is.null(model$mod_shift)) {
    ret$w_shift <- model$mod_shift$get_weights()
  }
  if (!is.null(model$mod_image)) {
    ret$w_image <- model$mod_image$get_weights()
  }
  return(ret)
}
