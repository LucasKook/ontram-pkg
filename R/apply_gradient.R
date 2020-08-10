#' Compute gradient contribution for exact response
#' @examples
#' mo <- ontram_polr(x_dim = 1L, y_dim = 5L, method = "logit")
#' x_train <- matrix(c(0.5, -0.5, 0), ncol = 1)
#' x_train <- tf$constant(x_train, dtype = "float32")
#' y_train <- matrix(c(0, 1, 0, 0, 0,
#'                     0, 0, 1, 0, 0,
#'                     0, 0, 0, 0, 1), nrow = 3, byrow = TRUE)
#' y_train <- tf$constant(y_train, dtype = "float32")
#' apply_gradient(x_train, y_train, mo, verbose = TRUE)
#' @export
apply_gradient <- function(x_train, y_train, model, img_train = NULL,
                           verbose = FALSE, response_varying = FALSE) {
  with(tf$GradientTape() %as% tape, {
    if (response_varying) {
      fwd_gamma <- model$mod_baseline(img_train)
    } else {
      fwd_gamma <- model$mod_baseline(matrix(1))
      fwd_gamma <- k_reshape(fwd_gamma, c(1L, model$y_dim - 1L))
    }
    fwd_beta <- NULL
    if (!is.null(x_train) & !is.null(model$mod_shift)) {
      fwd_beta <- model$mod_shift(x_train)
    }
    fwd_eta <- NULL
    if (!is.null(img_train) & !is.null(model$mod_image)) {
      fwd_eta <- model$mod_image(img_train)
    }
    nll <- compute_logLik(gammas = fwd_gamma, beta = fwd_beta, eta = fwd_eta,
                          y_train = y_train, distr = model$distr)
  })
  train_parms <- list(model$mod_baseline$trainable_variables)
  if (!is.null(model$mod_shift))
    train_parms <- append(train_parms, list(model$mod_shift$trainable_variables))
  if (!is.null(model$mod_image))
    train_parms <- append(train_parms, list(model$mod_image$trainable_variables))
  nabla <- tape$gradient(nll, train_parms)
  if (verbose)
    print(nabla)
  nparm <- length(train_parms)
  for (i in seq_len(nparm)) {
    model$optimizer$apply_gradients(
      purrr::transpose(list(nabla[[i]], train_parms[[i]]))
    )
  }
}
