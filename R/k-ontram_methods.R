
#' S3 methods for \code{k_ontram}
#' @method predict k_ontram
#' @export
predict.k_ontram <- function(object, x,
                             type = c("distribution", "density", "trafo",
                                      "baseline_only", "hazard", "cumhazard",
                                      "survivor", "odds", "terms"),
                             ...) {
  type <- match.arg(type)
  if ("k_ontram_ci" %in% class(object)) {
    class(object) <- class(object)[-2L]
  } else {
    class(object) <- class(object)[-1L]
  }
  preds <- predict(object, x = x, ... = ...)
  if (type == "terms")
    return(preds)
  K <- ncol(preds)
  baseline <- preds[, 1L:(K - 1L)]
  shift <- do.call("cbind", lapply(1L:(K - 1L), function(x) preds[, K]))
  trafo <- baseline - shift
  ccdf <- cbind(plogis(trafo), 1)
  cdf <- cbind(0, ccdf)
  pdf <- t(apply(cdf, 1, diff))
  surv <- 1 - ccdf
  haz <- pdf / (1 - ccdf)
  cumhaz <- - log(surv)
  odds <- ccdf / (1 - ccdf)

  ret <- switch(type,
                "distribution" = cdf,
                "density" = pdf,
                "trafo" = trafo,
                "baseline_only" = baseline,
                "hazard" = haz,
                "cumhazard" = cumhaz,
                "survivor" = surv,
                "odds" = odds)

  return(ret)
}

#' Simulate Responses
#' @method simulate k_ontram
#' @param object an object of class \code{\link{k_ontram}}.
#' @param x list of data matrices (including matrix containing 1 if model intercept is non-complex)
#' @param nsim number of simulations.
#' @param levels levels of simulated ordered responses.
#' @param seed random seed.
#' @examples
#' data(wine, package = "ordinal")
#' fm <- rating ~ temp + contact
#' y <- model.matrix(~ 0 + rating, data = wine)
#' x <- ontram:::.rm_int(model.matrix(fm, data = wine))
#' loss <- k_ontram_loss(ncol(y))
#'
#' mbl <- k_mod_baseline(ncol(y), name = "baseline")
#' msh <- mod_shift(ncol(x), name = "linear_shift")
#'
#' mo <- k_ontram(mbl, msh)
#' compile(mo, optimizer = optimizer_adam(learning_rate = 10^-4), loss = loss)
#' fit(mo, x = list(matrix(1, nrow = nrow(wine)), x), y = y, batch_size = nrow(wine), epoch = 10)
#' simulate(mo, x = list(matrix(1, nrow = nrow(wine)), x), nsim = 1)
#' @export
simulate.k_ontram <- function(object, x, nsim = 1, levels = NULL, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  pr <- predict(object, x = x, type = "density")
  if (is.null(levels)) {
    levels <- 1:ncol(pr)
  }
  ret <- apply(pr, 1, function(p) sample(levels, nsim, prob = p, replace = TRUE))
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

#' Set initial weights
#' @method warmstart k_ontram
#' @param object an object of class \code{\link{k_ontram}}.
#' @param thetas intercepts of a \code{\link[tram]{Polr}} model as a vector (used to initialize the weights of the baseline model).
#' @param betas shift terms of a \code{\link[tram]{Polr}} model as a vector (used to initialize the weights of the linear shift model).
#' @examples
#' library(tram)
#' set.seed(2021)
#' data(wine, package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' y <- model.matrix(~ 0 + rating, data = wine)
#' x <- ontram:::.rm_int(model.matrix(rating ~ temp + contact, data = wine))
#' im <- ontram:::.rm_int(model.matrix(rating ~ noise, data = wine))
#' loss <- k_ontram_loss(ncol(y))
#' mod_polr <- Polr(rating ~ temp + contact, data = wine)
#' mbl <- k_mod_baseline(ncol(y))
#' msh <- mod_shift(ncol(x))
#' mim <- keras_model_sequential() %>%
#'   layer_dense(units = 8, input_shape = 1L, activation = "relu") %>%
#'   layer_dense(units = 16, activation = "relu") %>%
#'   layer_dense(units = 1, use_bias = FALSE)
#' mo <- k_ontram(mbl, list(msh, mim))
#' mo <- warmstart(mo, thetas = coef(mod_polr, with_baseline = T)[1:4L],
#'                 which = "baseline only")
#' coef(mod_polr, with_baseline = TRUE)
#' ontram:::.to_theta(get_weights(mo$mod_baseline)[[1]])
#' @export
warmstart.k_ontram <- function(object, thetas = NULL, betas = NULL,
                               which = c("all", "baseline only", "shift only")) {
  which <- match.arg(which)
  K <- object$mod_baseline$output_shape[[2L]] + 1
  nshift <- length(object$list_of_shift_models)
  if (which == "all" || which == "baseline only") {
    gammas <- list(matrix(.to_gamma(thetas), nrow = 1, ncol = K-1))
    set_weights(object$mod_baseline, gammas)
  }
  if (which == "all" || which == "shift only") {
    if (object$list_of_shift_models$layers[[1]]$trainable & !is.null(betas)) {
      betas <- list(matrix(betas, nrow = length(betas), ncol = 1))
      if (nshift == 1L) {
        if (nrow(get_weights(object$list_of_shift_models)[[1]]) == nrow(betas[[1]])) {
          set_weights(object$list_of_shift_models, betas)
        }
      } else if (nshift >= 2L) {
        for (idx in 0:nshift - 1L) {
          if (nrow(get_weights(object$list_of_shift_models[[idx]])[[1]]) == nrow(betas[[1]]) &
              ncol(get_weights(object$list_of_shift_models[[idx]])[[1]]) == 1) {
            set_weights(object$list_of_shift_models[[idx]], betas)
            break
          }
        }
      }
    }
  }
  return(invisible(object))
}

#' Set initial weights
#' @method warmstart k_ontram_ci
#' @param thetas intercepts of a \code{\link[tram]{Polr}} model as a vector (are added to the last layer during training).
#' @param betas shift terms of a \code{\link[tram]{Polr}} model as a vector (used to initialize the weights of the simple shift model).
#' @section IMPORTANT:
#' \itemize{
#' \item{The warmstarted model has to be assigned to a new variable.}
#' }
#' @examples
#' library(tram)
#' set.seed(2021)
#' data(wine, package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' y <- model.matrix(~ 0 + rating, data = wine)
#' x <- ontram:::.rm_int(model.matrix(rating ~ temp + contact, data = wine))
#' im <- ontram:::.rm_int(model.matrix(rating ~ noise, data = wine))
#' loss <- k_ontram_loss(ncol(y))
#' mod_polr <- Polr(rating ~ temp + contact, data = wine)
#' msh <- mod_shift(ncol(x))
#' mbl <- keras_model_sequential() %>%
#'   layer_dense(units = 8, input_shape = 1L, activation = "relu") %>%
#'   layer_dense(units = 16, activation = "relu") %>%
#'   layer_dense(units = ncol(y) - 1, use_bias = FALSE) %>%
#'   layer_trafo_intercept()()
#' mo <- k_ontram(mbl, msh, complex_intercept = TRUE)
#' mo <- warmstart(mo, thetas = coef(mod_polr, with_baseline = TRUE)[1:4L],
#'                 betas = coef(mod_polr), which = "all")
#' @export
warmstart.k_ontram_ci <- function(object, thetas = NULL, betas = NULL,
                                  which = c("all", "baseline only", "shift only")) {
  which <- match.arg(which)
  K <- object$output_shape[[2L]]
  nshift <- length(object$list_of_shift_models)
  if (which == "all" || which == "baseline only") {
    gammas <- .to_gamma(thetas)
    mbl_copy <- clone_model(object$mod_baseline)
    pop_layer(mbl_copy) # remove lambda layer
    ll <- length(mbl_copy$layers)
    ll_activation <- mbl_copy$layers[[ll]]$activation
    mbl_new <- mbl_copy %>%
      .layer_add_gamma_tilde(gammas)() %>% # add gammas
      layer_trafo_intercept()()
    object <- k_ontram(mbl_new, object$list_of_shift_models,
                       complex_intercept = TRUE)
  }
  if (which == "all" || which == "shift only") {
    if (object$list_of_shift_models$layers[[1]]$trainable & !is.null(betas)) {
      betas <- list(matrix(betas, nrow = length(betas), ncol = 1))
      if (nshift == 1L) {
        if (nrow(get_weights(object$list_of_shift_models)[[1]]) == nrow(betas[[1]])) {
          set_weights(object$list_of_shift_models, betas)
        }
      } else if (nshift >= 2L) {
        for (idx in 0:nshift - 1L) {
          if (nrow(get_weights(object$list_of_shift_models[[idx]])[[1]]) == nrow(betas[[1]]) &
              ncol(get_weights(object$list_of_shift_models[[idx]])[[1]]) == 1) {
            set_weights(object$list_of_shift_models[[idx]], betas)
            break
          }
        }
      }
    }
  }
  return(invisible(object))
}

.layer_add_gamma_tilde <- function(gammas) {
  gammas <- k_constant(gammas)
  tf$keras$layers$Lambda(
    function(x) {
      tf$math$add(x, gammas)
    }
  )
}

