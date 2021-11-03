#' Get/Set model weights
#' @name weights_ontram
#' @aliases get_weights_ontram
#' @aliases set_weights_ontram
#' @rdname weights_ontram
#' @examples
#' set.seed(2021)
#' data("wine", package = "ordinal")
#' wine$noise <- rnorm(nrow(wine), sd = 0.3) + as.numeric(wine$rating)
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' im_train <- model.matrix(rating ~ noise, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' mo1 <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
#'                    method = "logit", n_batches = 10, epochs = 50)
#' fit_ontram(mo1, x_train = x_train, y_train = y_train)
#'
#' mbl <- keras_model_sequential() %>%
#'   layer_dense(units = 4, input_shape = 1L, use_bias = TRUE, activation = "tanh") %>%
#'   layer_dense(units = 4, use_bias = TRUE)
#' msh <- mod_shift(ncol(x_train))
#' mo2 <- ontram(mod_bl = mbl, mod_sh = msh, method = "logit", n_batches = 10,
#'               epochs = 20, x_dim = 1L, y_dim = ncol(y_train),
#'               response_varying = TRUE)
#'
#' w_mo1 <- get_weights_ontram(mo1, w_shift = TRUE)
#' get_weights_ontram(mo2, w_shift = TRUE)$w_shift
#' set_weights_ontram(mo2, weights = w_mo1)
#' get_weights_ontram(mo2)$w_shift
#' @param model an object of class \code{\link{ontram}}.
#' @param w_baseline logical. Whether weights of baseline model should be extracted.
#' @param w_shift logical. Whether weights of shift model should be extracted.
#' @param w_image logical. Whether weights of image model should be extracted.
#' @export
get_weights_ontram <- function(model, w_baseline = FALSE, w_shift = FALSE, w_image = FALSE) {
  if (w_baseline) {
    wbl <- model$mod_baseline$get_weights()
  } else {
    wbl <- NULL
  }
  if (w_shift) {
    wsh <- model$mod_shift$get_weights()
  } else {
    wsh <- NULL
  }
  if (w_image) {
    wim <- model$mod_image$get_weights()
  } else {
    wim <- NULL
  }
  ret <- list(w_baseline = wbl,
              w_shift = wsh,
              w_image = wim)
  return(ret)
}

#' @rdname weights_ontram
#' @param model an object of class \code{\link{ontram}}.
#' @param weights output of \code{\link{get_weights_ontram}} or list of similar structure;
#' lists with corresponding names ("w_baseline", "w_shift", "w_image") containing weights as arrays.
#' @export
set_weights_ontram <- function(model, weights) {
  if (!is.null(weights$w_baseline)) {
    model$mod_baseline$set_weights(weights = weights$w_baseline)
  }
  if (!is.null(weights$w_shift)) {
    model$mod_shift$set_weights(weights = weights$w_shift)
  }
  if (!is.null(weights$w_image)) {
    model$mod_image$set_weights(weights = weights$w_image)
  }
}
