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

#' Simulate Responses
#' @method simulate ontram
#' @examples
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' x_valid <- x_train[1:20,]
#' y_valid <- y_train[1:20,]
#' mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
#'                   method = "logit", n_batches = 10, epochs = 50)
#' fit_ontram(mo, x_train = x_train, y_train = y_train)
#' simulate(mo, nsim = 1, x = x_valid, y = y_valid)
#' @export
simulate.ontram <- function(object, nsim = 1, x = NULL, y, im = NULL, levels = 1:ncol(y), seed = NULL, ...) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  pr <- predict(object, x = x, y = y, im = im)
  ret <- apply(pr$pdf, 1, function(p) sample(levels, nsim, prob = p, replace = TRUE))
  if (nsim > 1) {
    tmp <- vector(mode = "list", length = nsim)
    for (i in 1:nsim) {
      tmp[[i]] <- ordered(ret[i, ], levels = levels)
    }
    ret <- tmp
  } else {
    ret <- ordered(ret, levels = levels)
  }
  return(ret)
}

# generic function
#' @export
warmstart <- function(object, ...) {
  UseMethod("warmstart")
}

#' Set initial weights
#' @method warmstart ontram
#' @param object an object of class \code{\link{ontram}}.
#' @param object_w an object of class \code{\link[tram]{Polr}} from which model weights are taken.
#' @examples
#' library(tram)
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' mp <- Polr(fml, data = wine)
#' mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train))
#' coef(mp, with_baseline = T)
#' warmstart(mo, mp, "all")
#' mod_weights(mo)
#' @export
warmstart.ontram <- function(object, object_w, which = c("all", "baseline only", "shift only")) {
  stopifnot(which %in% c("all", "baseline only", "shift only"))
  K <- object$y_dim
  x_dim <- object$x_dim
  w <- vector(mode = "list", length = 2)
  names(w) <- c("w_baseline", "w_shift")
  w$w_baseline <- list(matrix(ontram:::.to_gamma(coef(object_w, with_baseline = T)[1:K-1]),
                              nrow = 1, ncol = K-1))
  if (!is.null(coef(object_w))) {
    w$w_shift <- list(matrix(coef(object_w),
                             nrow = x_dim, ncol = 1))
  }
  if (which %in% "baseline only") {
    object$mod_baseline$set_weights(w$w_baseline)
  }
  if (which %in% "shift only") {
    object$mod_shift$set_weights(w$w_shift)
  }
  if (which %in% "all") {
    object$mod_baseline$set_weights(w$w_baseline)
    object$mod_shift$set_weights(w$w_shift)
  }
  return(invisible(object))
}

#' Set initial weights
#' @method warmstart ontram_rv
#' @param object an object of class \code{ontram_rv}.
#' @param object_w an object of class \code{\link[tram]{Polr}}, \code{ontram_rv} or \code{\link{ontram}}
#' from which model weights are taken.
#' @examples
#' library(tram)
#' set.seed(2021)
#' data("wine", package = "ordinal")
#' wine$noise <- rnorm(nrow(wine), sd = 0.3) + as.numeric(wine$rating)
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L, drop = FALSE]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' im_train <- model.matrix(rating ~ noise, data = wine)[, -1L, drop = FALSE]
#'
#' mp <- Polr(fml, data = wine)
#' mbl1 <- keras_model_sequential() %>%
#'          layer_dense(units = 4, input_shape = 1L, activation = "tanh") %>%
#'          layer_dense(ncol(y_train) - 1)
#' msh1 <- mod_shift(ncol(x_train))
#' mo_rv1 <- ontram(mod_bl = mbl1, mod_sh = msh1, method = "logit",
#'                  x_dim = ncol(x_train), y_dim = ncol(y_train),
#'                  response_varying = TRUE)
#' fit_ontram(mo_rv1, x_train = x_train, y_train = y_train, img_train = im_train)
#'
#' mbl2 <- keras_model_sequential() %>%
#'          layer_dense(units = 4, input_shape = 1L, activation = "tanh") %>%
#'          layer_dense(ncol(y_train) - 1)
#' msh2 <- mod_shift(ncol(x_train))
#' mo_rv2 <- ontram(mod_bl = mbl2, mod_sh = msh2, method = "logit",
#'                  x_dim = ncol(x_train), y_dim = ncol(y_train),
#'                  response_varying = TRUE)
#' mod_weights(mo_rv2)
#' warmstart(mo_rv2, mo_rv1, "baseline only")
#' mod_weights(mo_rv2)
#' coef(mp)
#' warmstart(mo_rv2, mp, "shift only")
#' mod_weights(mo_rv2)

#' @export
warmstart.ontram_rv <- function(object, object_w, which = c("all", "baseline only", "shift only")) {
  stopifnot(which %in% c("all", "baseline only", "shift only"))
  if (identical(mod_weights(object), mod_weights(object_w))) {
    stop("`object` and `object_w` are identical.")
  }
  K <- object$y_dim
  x_dim <- object$x_dim
  w <- vector(mode = "list", length = 2)
  names(w) <- c("w_baseline", "w_shift")
  if ("tram" %in% class(object_w)) {
    w$w_shift <- list(matrix(coef(object_w),
                             nrow = x_dim, ncol = 1))
    if (which %in% "baseline only") {
      stop("If which = `baseline only` object_w must be of class `ontram_rv`.")
    }
    if (which %in% "shift only") {
      object$mod_shift$set_weights(w$w_shift)
    }
    if (which %in% "all") {
      stop("If which = `all` object_w must be of class `ontram_rv`.")
    }
  }
  if ("ontram" %in% class(object_w)) {
    w$w_baseline <- object_w$mod_baseline$get_weights()
    if (!is.null(object_w$mod_shift)) {
      w$w_shift <- object_w$mod_shift$get_weights()
    }
    if (which %in% "baseline only") {
      object$mod_baseline$set_weights(w$w_baseline)
    }
    if (which %in% "shift only") {
      object$mod_shift$set_weights(w$w_shift)
    }
    if (which %in% "all") {
      object$mod_baseline$set_weights(w$w_baseline)
      object$mod_shift$set_weights(w$w_shift)
    }
  }
  return(invisible(object))
}

# generic function
#' @export
mod_weights <- function(model, ...) {
 UseMethod("mod_weights")
}

#' Extract model weights
#' @method mod_weights ontram
#' @examples
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train))
#' mod_weights(mo)
#' @export
mod_weights.ontram <- function(model) {
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

