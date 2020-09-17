#' Function for coef for simple ontram
#' @method coef ontram
#' @examples
#' mod <- ontram_polr(2L, 5L)
#' coef(mod, with_baseline = TRUE)
#' @export
coef.ontram <- function(object, with_baseline = FALSE, ...) {
  if ("ontram_rv" %in% class(object)) {
    class(object) <- class(object)[-1L]
    coef(object, with_baseline = FALSE, ... = ...)
  }
  gammas <- object$mod_baseline$get_weights()[[1L]]
  cfx <- .to_theta(c(gammas))
  cfs <- c(object$mod_shift$get_weights()[[1L]])
  if (!with_baseline)
    return(cfs)
  return(c(cfx, cfs))
}

#' Function for predicting cdf, pdf and class and compute logLik
#' @method predict ontram
#' @export
predict.ontram <- function(model, x = NULL, y, im = NULL) {
  y <- tf$constant(y, dtype = "float32")
  gammas <- model$mod_baseline(matrix(1))
  gammas <- k_reshape(gammas, c(1L, model$y_dim - 1L))
  thetas <- gamma_to_theta(gammas)
  if (!is.null(x)) {
    x <- tf$constant(x, dtype = "float32")
    betas <- model$mod_shift(x)
  } else {
    betas <- tf$zeros_like(thetas)
  }
  if (!is.null(im)) {
    if (!is.list(im))
      im <- tf$constant(im, dtype = "float32")
    etas <- model$mod_image(im)
  } else {
    etas <- tf$zeros_like(thetas)
  }
  distr <- model$distr
  probs <- distr(thetas - betas - etas)$numpy()
  dens <- t(apply(probs, 1, diff))
  cls <- apply(dens, 1, which.max)
  nll <- compute_logLik(gammas, betas, etas, y)$numpy()
  return(list(cdf = probs, pdf = dens, response = cls, negLogLik = nll))
}

#' Function for predicting cdf, pdf and class and compute logLik
#' @method predict ontram_rv
#' @export
predict.ontram_rv <- function(model, x, y, im = NULL) {
  y <- tf$constant(y, dtype = "float32")
  gammas <- model$mod_baseline(im)
  thetas <- gamma_to_theta(gammas)
  if (!is.null(x)) {
    x <- tf$constant(x, dtype = "float32")
    betas <- model$mod_shift(x)
  } else {
    betas <- tf$zeros_like(thetas)
  }
  distr <- model$distr
  probs <- distr(thetas - betas)$numpy()
  dens <- t(apply(probs, 1, diff))
  cls <- apply(dens, 1, which.max)
  if (!is.null(x)) {
    betas <- model$mod_shift(x)
  } else {
    betas <- NULL
  }
  nll <- compute_logLik(gammas, betas, NULL, y)$numpy()
  return(list(cdf = probs, pdf = dens, response = cls, negLogLik = nll))
}

#' Plot ontram history
#' @method plot ontram_history
#' @export
plot.ontram_history <- function(object, col_train = "blue", col_test = "red", ...) {
  parms <- list(...)
  epoch <- seq_len(length(object$train_loss))
  if ("ylim" %in% names(parms)) {
    plot(epoch, object$train_loss, type = "l", col = col_train,
         ylab = "negative logLik", ... = ...)
  } else {
    all_dat <- do.call("rbind", object)
    mylim <- range(all_dat)
    plot(epoch, object$train_loss, type = "l", col = col_train, ylim = mylim,
         ylab = "negative logLik", ... = ...)
  }
  lines(epoch, object$test_loss, col = col_test)
  legend("topright", legend = c("Train", "Test"), col = c(col_train, col_test),
         bty = "n", lwd = 2)
}

# Function for saving ontram models
save_model.ontram <- function(object, filename, ...) {
  nm_theta <- paste0(filename, "_theta.h5")
  nm_beta <- paste0(filename, "_beta.h5")
  nm_rest <- paste0(filename, "_r.Rds")
  rest <- list(x_dim = object$x_dim,
               y_dim = object$y_dim,
               n_batches = object$n_batches,
               epochs = object$epochs)
  save(rest, file = nm_rest)
  save_model_hdf5(object$mod_baseline, nm_theta)
  save_model_hdf5(object$mod_shift, nm_beta)
}

# Function for loading ontram models
load_model.ontram <- function(filename, ...) {
  nm_theta <- paste0(filename, "_theta.h5")
  nm_beta <- paste0(filename, "_beta.h5")
  nm_rest <- paste0(filename, "_r.Rds")
  load(nm_rest)
  mt <- load_model_hdf5(nm_theta)
  mb <- load_model_hdf5(nm_beta)
  ret <- append(rest, list(mod_baseline = mt, mod_shift = mb,
                           optimizer = tf$keras$optimizers$Adam(learning_rate = 0.001),
                           distr = tf$sigmoid))
  class(ret) <- "ontram"
  return(ret)
}
