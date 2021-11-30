#' Keras interface to ONTRAMs
#' @examples
#' library(tram)
#' set.seed(2021)
#' mbl <- k_mod_baseline(5L, name = "baseline")
#' msh <- mod_shift(2L, name = "linear_shift")
#' mim <- mod_shift(1L, name = "complex_shift")
#' m <- k_ontram(mbl, list(msh, mim))
#'
#' data("wine", package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' X <- .rm_int(model.matrix(~ temp + contact, data = wine))
#' Y <- model.matrix(~ 0 + rating, data = wine)
#' Z <- .rm_int(model.matrix(~ noise, data = wine))
#' INT <- matrix(1, nrow = nrow(wine))
#'
#' m(list(INT, X, Z))
#' loss <- k_ontram_loss(ncol(Y))
#' compile(m, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
#' fit(m, x = list(INT, X, Z), y = Y, batch_size = nrow(wine), epoch = 10,
#'     view_metrics = FALSE)
#'
#' idx <- 8
#' loss(Y[idx, , drop = FALSE], m(list(INT[idx, , drop = FALSE],
#'      X[idx, , drop = FALSE], Z[idx, , drop = FALSE])))
#'
#' tm <- Polr(rating ~ temp + contact + noise, data = wine)
#' logLik(tm, newdata = wine[idx,])
#'
#' tmp <- get_weights(m)
#' tmp[[1]][] <- .to_gamma(coef(as.mlt(tm))[1:4])
#' tmp[[2]][] <- coef(tm)[1:2]
#' tmp[[3]][] <- coef(tm)[3]
#' set_weights(m, tmp)
#'
#' loss(k_constant(Y), m(list(INT, X, Z)))
#' - logLik(tm)
#'
#' ## Complex intercept ##
#'
#' mim2 <- keras_model_sequential() %>%
#'    layer_dense(units = 4, input_shape = 1L, activation = "relu") %>%
#'    layer_dense(units = ncol(Y) - 1, use_bias = FALSE)
#'
#' mbl2 <- k_mod_baseline(ncol(Y), mod_complex = mim2)
#' m2 <- k_ontram(mbl2)
#' compile(m2, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
#' fit(m2, x = list(INT, Z), y = Y, batch_size = nrow(wine), epoch = 10,
#'     view_metrics = FALSE)
#'
#' get_weights(m2$mod_baseline[[0]])
#' warmstart(m2, tm, "baseline only")
#' get_weights(m2$mod_baseline[[0]])
#' @export
# k_ontram <- function(
#   mod_baseline,
#   list_of_shift_models = NULL,
#   ...
# ) {
#   if (is.null(list_of_shift_models)) {
#     list_of_shift_models <- keras_model_sequential() %>%
#       layer_dense(units = 1L, input_shape = c(1L),
#                   kernel_initializer = initializer_zeros(),
#                   trainable = FALSE)
#   }
#   nshift <- length(list_of_shift_models)
#   if (nshift == 1L) {
#     shift_in <- list_of_shift_models$input
#     shift_out <- list_of_shift_models$output
#   } else if (nshift >= 2L) {
#     shift_in <- lapply(list_of_shift_models, function(x) x$input)
#     shift_out <- lapply(list_of_shift_models, function(x) x$output) %>%
#       layer_add()
#   }
#   inputs <- list(mod_baseline$input, shift_in)
#   outputs <- list(mod_baseline$output, shift_out)
#   m <- keras_model(inputs = inputs, outputs = layer_concatenate(outputs))
#   m$mod_baseline <- mod_baseline
#   m$list_of_shift_models <- list_of_shift_models
#   class(m) <- c("k_ontram", class(m))
#   return(m)
# }
k_ontram <- function(
  mod_baseline,
  list_of_shift_models = NULL,
  ...
) {
  nbaseline <- length(mod_baseline)
  nshift <- length(list_of_shift_models)
  if (nbaseline == 1L) {
    baseline_in <- mod_baseline$input
    baseline_out <- mod_baseline$output %>%
      layer_trafo_intercept()()
  } else if (nbaseline == 2) {
    baseline_in <- lapply(mod_baseline, function(x) x$input)
    baseline_out <- lapply(mod_baseline, function(x) x$output) %>%
      layer_add() %>%
       layer_trafo_intercept()()
  }
  if (is.null(list_of_shift_models)) {
    list_of_shift_models <- keras_model_sequential() %>%
      layer_dense(units = 1L, input_shape = c(1L),
                  kernel_initializer = initializer_zeros(),
                  use_bias = FALSE, trainable = FALSE)
    shift_in <- list_of_shift_models$input
    shift_out <- list_of_shift_models$output
  }
  if (nshift == 1L) {
    shift_in <- list_of_shift_models$input
    shift_out <- list_of_shift_models$output
  } else if (nshift >= 2L) {
    shift_in <- lapply(list_of_shift_models, function(x) x$input)
    shift_out <- lapply(list_of_shift_models, function(x) x$output) %>%
      layer_add()
  }
  inputs <- list(baseline_in, shift_in)
  outputs <- list(baseline_out, shift_out)
  m <- keras_model(inputs = inputs, outputs = layer_concatenate(outputs))
  m$mod_baseline <- mod_baseline
  m$list_of_shift_models <- list_of_shift_models
  if (nbaseline == 1) {
    class(m) <- c("k_ontram", class(m))
  } else if (nbaseline == 2) {
    class(m) <- c("k_ontram_rv", "k_ontram", class(m))
  }
  return(m)
}

#' Another keras implementation of the ontram loss
#' @examples
#' y_true <- k_constant(matrix(c(1, 0, 0, 0, 0), nrow = 1L))
#' loss <- k_ontram_loss(ncol(y_true))
#' loss(y_true, m$output)
#' debugonce(loss)
#' loss(k_constant(Y), m(list(INT, X, Z)))
#' @export
k_ontram_loss <- function(K) {
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = TRUE]
    shifts <- y_pred[, K, drop = TRUE]
    yu <- y_true[, 1L:(K - 1L), drop = FALSE]
    yl <- y_true[, 2L:K, drop = FALSE]
    upr <- k_sum(tf$multiply(yu, intercepts), axis = 0L) - shifts
    lwr <- k_sum(tf$multiply(yl, intercepts), axis = 0L) - shifts
    t1 <- y_true[, 1L, drop = TRUE]
    tK <- y_true[, K, drop = TRUE]
    lik <- t1 * k_sigmoid(upr) + tK * (1 - k_sigmoid(lwr)) +
      (1 - t1) * (1 - tK) * (k_sigmoid(upr) - k_sigmoid(lwr))
    - k_sum(k_log(lik))
  }
}

#' Layer for transforming raw intercepts
#' @examples
#' layer_trafo_intercept()
#' @export
layer_trafo_intercept <- function() {
  tf$keras$layers$Lambda(
    function(x) {
      w1 <- x[, 1L, drop = FALSE]
      wrest <- tf$math$exp(x[, 2L:x$shape[[2]], drop = FALSE])
      tf$cumsum(k_concatenate(list(w1, wrest), axis = 0L), axis = 1L)
    }
  )
}

#' keras mbl
#' @examples
#' mbl <- k_mod_baseline(5)
#' mbl(matrix(1))
#' @export
# k_mod_baseline <- function(K, ...) {
#   keras_model_sequential() %>%
#     layer_dense(units = K - 1L, input_shape = 1L, use_bias = FALSE,
#                 ... = ...) %>%
#     layer_trafo_intercept()()
# }
k_mod_baseline <- function(K, mod_complex = NULL, ...) {
  if (is.null(mod_complex)) {
    keras_model_sequential() %>%
      layer_dense(units = K - 1L, input_shape = 1L, use_bias = FALSE,
                  ... = ...)
  } else {
   mod_gamma_tilde <- keras_model_sequential() %>%
      layer_dense(units = K - 1L, input_shape = c(1L),
                  kernel_initializer = initializer_zeros(),
                  use_bias = FALSE, trainable = FALSE)
   mod_epsilon <- mod_complex
   list(mod_gamma_tilde, mod_epsilon)
  }
}

#' S3 methods for \code{k_ontram}
#' @method predict k_ontram
#' @export
# predict.k_ontram <- function(object, x,
#                              type = c("distribution", "density", "trafo",
#                                       "baseline_only", "hazard", "cumhazard",
#                                       "survivor", "odds"),
#                              ...) {
#   type <- match.arg(type)
#   class(object) <- class(object)[-1L]
#   preds <- predict(object, x = x, ... = ...)
#   K <- ncol(preds)
#   baseline <- preds[, 1L:(K - 1L)]
#   shift <- do.call("cbind", lapply(1L:(K - 1L), function(x) preds[, K]))
#   trafo <- baseline - shift
#   ccdf <- cbind(plogis(trafo), 1)
#   cdf <- cbind(0, ccdf)
#   pdf <- t(apply(cdf, 1, diff))
#   surv <- 1 - ccdf
#   haz <- pdf / (1 - ccdf)
#   cumhaz <- - log(surv)
#   odds <- ccdf / (1 - ccdf)
#
#   ret <- switch(type,
#                 "distribution" = cdf,
#                 "density" = pdf,
#                 "trafo" = trafo,
#                 "baseline_only" = baseline,
#                 "hazard" = haz,
#                 "cumhazard" = cumhaz,
#                 "survivor" = surv,
#                 "odds" = odds)
#
#   return(ret)
# }
predict.k_ontram <- function(object, x,
                             type = c("distribution", "density", "trafo",
                                      "baseline_only", "hazard", "cumhazard",
                                      "survivor", "odds"),
                             ...) {
  type <- match.arg(type)
  if ("k_ontram_rv" %in% class(object)) {
    class(object) <-  class(object)[-2L]
  } else {
    class(object) <- class(object)[-1L]
  }
  preds <- predict(object, x = x, ... = ...)
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
#' @param object_w an object of class \code{\link[tram]{Polr}} or \code{\link{k_ontram}} from which weights are taken.
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
#' mbl <- k_mod_baseline(ncol(y), name = "baseline")
#' msh <- mod_shift(ncol(x), name = "shift")
#' mim <- keras_model_sequential() %>%
#'   layer_dense(units = 8, input_shape = 1L, activation = "relu") %>%
#'   layer_dense(units = 16, activation = "relu") %>%
#'   layer_dense(units = 1, use_bias = FALSE)
#' mo <- k_ontram(mbl, list(msh, mim))
#' warmstart(mo, mod_polr, which = "baseline only")
#'
#' compile(mo, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2))
#' fit(mo, x = list(matrix(1, nrow = nrow(wine)), x, im), y = y)
#' mbl2 <- k_mod_baseline(ncol(y))
#' mim2 <- keras_model_sequential() %>%
#'   layer_dense(units = 8, input_shape = 1L, activation = "relu") %>%
#'   layer_dense(units = 16, activation = "relu") %>%
#'   layer_dense(units = 1, use_bias = FALSE)
#' mo2 <- k_ontram(mbl2, mim2)
#' warmstart(mo2, mo, which = "all")
#' @export
# warmstart.k_ontram <- function(object, object_w, which = c("all", "baseline only", "shift only")) {
#   which <- match.arg(which)
#   K <- object$output_shape[[2L]]
#   nshift <- length(object$list_of_shift_models)
#   if ("tram" %in% class(object_w)) {
#     if (which == "all" || which == "baseline only") {
#       w_baseline <- list(matrix(ontram:::.to_gamma(coef(object_w, with_baseline = T)[1:K-1]),
#                                 nrow = 1, ncol = K-1))
#       set_weights(object$mod_baseline, w_baseline)
#     }
#     if (which == "all" || which == "shift only") {
#       if (!is.null(coef(object_w))) {
#         w_shift <- list(matrix(coef(object_w),
#                                nrow = length(coef(object_w)), ncol = 1))
#         if (nshift == 1L) {
#           set_weights(object$list_of_shift_models, w_shift)
#         } else if (nshift >= 2L) {
#           for (idx in 0:nshift - 1L) {
#             if (nrow(get_weights(object$list_of_shift_models[[idx]])[[1]]) == nrow(w_shift[[1]])) {
#               set_weights(object$list_of_shift_models[[idx]], w_shift)
#             }
#           }
#         }
#       }
#     }
#   } else if ("k_ontram" %in% class(object_w)) {
#     if (which == "all" || which == "baseline only") {
#       w_baseline <- get_weights(object_w$mod_baseline)
#       set_weights(object$mod_baseline, w_baseline)
#     }
#     if (which == "all" || which == "shift only"){
#       if (nshift == 1L) {
#         if (length(object_w$list_of_shift_models) == 1L) {
#           w_shift <- get_weights(object_w$list_of_shift_models)
#           set_weights(object$list_of_shift_models, w_shift)
#         } else {
#           for (idx in 0:length(object_w$list_of_shift_models) - 1L) {
#             w_shift_old <- get_weights(object$list_of_shift_models)
#             w_shift <- get_weights(object_w$list_of_shift_models[[idx]])
#             if (length(unlist(w_shift_old)) == length(unlist(w_shift))) {
#               set_weights(object$list_of_shift_models, w_shift)
#             }
#           }
#         }
#       } else if (nshift >= 2L) {
#         for (idx in 0:nshift - 1L) {
#           w_shift_old <- get_weights(object$list_of_shift_models[[idx]])
#           if (length(object_w$list_of_shift_models) == 1L) {
#             w_shift <- get_weights(object_w$list_of_shift_models)
#               if (length(unlist(w_shift_old)) == length(unlist(w_shift))) {
#                 set_weights(object$list_of_shift_models[[idx]], w_shift)
#               }
#           } else {
#               for (i in 0:length(object_w$list_of_shift_models) - 1L) {
#                 w_shift <- get_weights(object_w$list_of_shift_models[[i]])
#                 if (length(unlist(w_shift_old)) == length(unlist(w_shift))) {
#                   set_weights(object$list_of_shift_models[[idx]], w_shift)
#                 }
#               }
#           }
#         }
#       }
#     }
#   }
#   return(invisible(object))
# }
warmstart.k_ontram <- function(object, object_w, which = c("all", "baseline only", "shift only")) {
  which <- match.arg(which)
  K <- object$output_shape[[2L]]
  nbaseline <- length(object$mod_baseline)
  nshift <- length(object$list_of_shift_models)
  if ("Polr" %in% class(object_w)) {
    if (which == "all" || which == "baseline only") {
      w_baseline <- list(matrix(ontram:::.to_gamma(coef(object_w, with_baseline = T)[1:K-1]),
                                nrow = 1, ncol = K-1))
      if (nbaseline == 1L) {
        set_weights(object$mod_baseline, w_baseline)
      }
      else if (nbaseline == 2L) {
        set_weights(object$mod_baseline[[0]], w_baseline)
      }
    }
    if (which == "all" || which == "shift only") {
      if (!is.null(coef(object_w))) {
        w_shift <- list(matrix(coef(object_w),
                               nrow = length(coef(object_w)), ncol = 1))
        if (nshift == 1L) {
          set_weights(object$list_of_shift_models, w_shift)
        } else if (nshift >= 2L) {
          for (idx in 0:nshift - 1L) {
            if (nrow(get_weights(object$list_of_shift_models[[idx]])[[1]]) == nrow(w_shift[[1]])) {
              set_weights(object$list_of_shift_models[[idx]], w_shift)
              break
            }
          }
        }
      }
    }
  } else if ("k_ontram" %in% class(object_w)) {
    if (which == "all" || which == "baseline only") {
      if (nbaseline == 1L) {
        w_baseline <- get_weights(object_w$mod_baseline)
        set_weights(object$mod_baseline, w_baseline)
      } else if (nbaseline == 2L) {
        w_baseline <- get_weights(object_w$mod_baseline[[1L]])
        set_weights(object$mod_baseline[[1L]], w_baseline)
      }
    }
    if (which == "all" || which == "shift only"){
      if (nshift == 1L) {
        if (length(object_w$list_of_shift_models) == 1L) {
          w_shift <- get_weights(object_w$list_of_shift_models)
          set_weights(object$list_of_shift_models, w_shift)
        } else {
          for (idx in 0:length(object_w$list_of_shift_models) - 1L) {
            w_shift_old <- get_weights(object$list_of_shift_models)
            w_shift <- get_weights(object_w$list_of_shift_models[[idx]])
            if (length(unlist(w_shift_old)) == length(unlist(w_shift))) {
              set_weights(object$list_of_shift_models, w_shift)
              break
            }
          }
        }
      } else if (nshift >= 2L) {
        for (idx in 0:nshift - 1L) {
          w_shift_old <- get_weights(object$list_of_shift_models[[idx]])
          if (length(object_w$list_of_shift_models) == 1L) {
            w_shift <- get_weights(object_w$list_of_shift_models)
            if (length(unlist(w_shift_old)) == length(unlist(w_shift))) {
              set_weights(object$list_of_shift_models[[idx]], w_shift)
              break
            }
          } else {
            for (i in 0:length(object_w$list_of_shift_models) - 1L) {
              w_shift <- get_weights(object_w$list_of_shift_models[[i]])
              if (length(unlist(w_shift_old)) == length(unlist(w_shift))) {
                set_weights(object$list_of_shift_models[[idx]], w_shift)
                break
              }
            }
          }
        }
      }
    }
  }
  return(invisible(object))
}

# warmstart.k_ontram <- function(object, object_w, which = c("all", "baseline only", "shift only")) {
#   stopifnot("Polr" %in% class(object_w))
#   which <- match.arg(which)
#   K <- object$output_shape[[2]]
#   n_layers <- length(object$get_weights())
#   w_new <- get_weights(object)
#   if (which == "all") {
#     w_new[[1]] <- matrix(ontram:::.to_gamma(coef(object_w, with_baseline = T)[1:K-1]),
#                          nrow = 1, ncol = K-1)
#     if (!is.null(coef(object_w))) {
#       for (idx in 2:n_layers) { # not optimal
#         if (ncol(w_new[[idx]]) == 1 && nrow(w_new[[idx]]) == length(coef(object_w))) {
#           w_new[[idx]] <- matrix(coef(object_w),
#                                  nrow = length(coef(object_w)), ncol = 1)
#         }
#       }
#     }
#   } else if (which == "baseline only") {
#     w_new[[1]] <- matrix(ontram:::.to_gamma(coef(object_w, with_baseline = T)[1:K-1]),
#                          nrow = 1, ncol = K-1)
#   }
#   else if (which == "shift only") {
#     for (idx in 2:n_layers) {
#       if (ncol(w_new[[idx]]) == 1 && nrow(w_new[[idx]]) == length(coef(object_w))) {
#         w_new[[idx]] <- matrix(coef(object_w),
#                                nrow = length(coef(object_w)), ncol = 1)
#       }
#     }
#   }
#   object$set_weights(w_new)
#   return(invisible(object))
# }

# layer_names <- numeric(length(object$layers))
# for (idx in 1:length(object$layers)) {
#   layer_names[idx] <- object$layers[[idx]]$name
# }

# else if("ontram_k" %in% class(object_w)) {
#   n_layers <- length(get_weights(object))
#   # complex intercept, no shift
#   if ((w_old[[n_layers]] == 0) && (w_old[[n_layers - 1]] == 0)) {
#     for (idx in 1:(n_layers - 2)) {
#       w_new$baseline[[idx]] <- get_weights(object_w)[[idx]]
#     }
#   }
#   # complex intercept, linear shift missing
#   # simple intercept complex shift missing
# }

