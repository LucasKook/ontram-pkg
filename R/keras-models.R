#' @export
k_polr <- function(y_dim, x_dim) {
  mbl <- mod_baseline(y_dim)
  msh <- mod_shift(x_dim)
  mim <- keras_model_sequential() %>%
    layer_dense(units = 1L, use_bias = FALSE, trainable = FALSE,
                kernel_initializer = initializer_zeros(), input_shape = 1L)
  m <- keras_model(inputs = list(mbl$input, msh$input, mim$input),
                   outputs = layer_concatenate(list(mbl$output, msh$output, mim$output)))
  return(m)
}
