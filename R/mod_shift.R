#' Sequential model for the shift function
#' @description Fully connected neural network with input dimension p and
#'     output dimension 1
#' @return Returns a model of type \code{\link[keras]{keras_mod_sequential}}
#' @importFrom keras keras_model_sequential layer_dense
#' @examples
#' m <- mod_shift(5L)
#' @export
mod_shift <- function(x_dim) {
  mod <- keras_model_sequential() %>%
    layer_dense(units = 1, input_shape = x_dim, use_bias = FALSE)
  return(mod)
}
