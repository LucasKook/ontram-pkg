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
plot.ontram_history <- function(object, col_train = "blue", col_test = "red", add_best = FALSE, ...) {
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
  if (add_best) {
    if (is.null(object$epoch_best)) stop("`epoch_best` not found.")
    abline(v = object$epoch_best, lty = 2)
  }
  legend("topright", legend = c("Train", "Test"), col = c(col_train, col_test),
         bty = "n", lwd = 2)
}

#' Function for saving ontram models
#' @export
save_model_ontram <- function(object, filename, ...) {
  nm_theta <- paste0(filename, "_theta.h5")
  nm_beta <- paste0(filename, "_beta.h5")
  nm_eta <- paste0(filename, "_eta.h5")
  nm_rest <- paste0(filename, "_r.Rds")
  rest <- list(x_dim = object$x_dim,
               y_dim = object$y_dim,
               n_batches = object$n_batches,
               epochs = object$epochs)
  save(rest, file = nm_rest)
  save_model_hdf5(object$mod_baseline, nm_theta)
  if (!is.null(object$mod_shift)) {
    save_model_hdf5(object$mod_shift, nm_beta)
  }
  if (!is.null(object$mod_image)) {
    save_model_hdf5(object$mod_image, nm_eta)
  }
}

#' Function for loading ontram models
#' @export
load_model_ontram <- function(filename, ...) {
  nm_theta <- paste0(filename, "_theta.h5")
  nm_beta <- paste0(filename, "_beta.h5")
  nm_eta <- paste0(filename, "_eta.h5") #ag: added
  nm_rest <- paste0(filename, "_r.Rds")
  load(nm_rest)
  mt <- load_model_hdf5(nm_theta)
  if (file.exists(nm_beta)) { #ag: model is only loaded when it exists
    mb <- load_model_hdf5(nm_beta)
  } else {
    mb <- NULL
  }
  if (file.exists(nm_eta)) {
    me <- load_model_hdf5(nm_eta) #ag: model is only loaded when it exists
  } else {
    me <- NULL
  }
  ret <- append(rest, list(mod_baseline = mt, mod_shift = mb, mod_image = me,
                           optimizer = tf$keras$optimizers$Adam(learning_rate = 0.001),
                           distr = tf$sigmoid))
  class(ret) <- "ontram"
  return(ret)
}

#' Function for saving ontram history
#' @export
save_ontram_history <- function(object, filepath) {
  write.table(data.frame(matrix(unlist(object[1:2]), nrow = 2, byrow = TRUE,
                                dimnames = list(c("train_loss", "test_loss"), NULL))),
              file = filepath, sep = ",", row.names = TRUE, col.names = FALSE)
  if (length(object) > 2) {
    write.table(object$epoch_best, file = filepath, sep = ",",
                row.names = "epoch_best", col.names = FALSE,
                append = TRUE)
  }
}

#' Function for loading ontram history
#' @export
load_ontram_history <- function(filepath) {
  df <- read.csv(filepath, header = FALSE)
  rownames(df) <- df[, 1]
  df <- df[, -1L]
  history <- list(train_loss = c(), test_loss = c())

  if (nrow(df) > 2) {
    history <- c(history, list(epoch_best = c()))
    history$epoch_best <- df[3,1]
  }
  history$train_loss <- as.numeric(df[1, ])
  history$test_loss <- as.numeric(df[2, ])
  class(history) <- "ontram_history"
  return(history)
}

