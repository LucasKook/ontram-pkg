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
simulate.ontram <- function(object, nsim = 1, seed = NULL, x = NULL, y, im = NULL, ...) {
  if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
    runif(1)
  if (is.null(seed))
    RNGstate <- get(".Random.seed", envir = .GlobalEnv)
  else {
    R.seed <- get(".Random.seed", envir = .GlobalEnv)
    set.seed(seed)
    RNGstate <- structure(seed, kind = as.list(RNGkind()))
    on.exit(assign(".Random.seed", R.seed, envir = .GlobalEnv))
  }
  pr <- predict(object, x = x, y = y, im = im)
  ret <- apply(pr$pdf, 1, function(p) sample(length(p), nsim, prob = p, replace = TRUE))
  if (nsim > 1) {
    tmp <- vector(mode = "list", length = nsim)
    for (i in 1:nsim) {
      tmp[[i]] <- ordered(ret[i, ])
    }
    ret <- tmp
  } else {
    ret <- ordered(ret)
  }
  return(ret)
}
