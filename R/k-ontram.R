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
#' cent <- metric_ontram_crossent(ncol(Y))
#' compile(m, loss = loss, optimizer = optimizer_adam(lr = 1e-2, decay = 0.001),
#' metrics = c(cent))
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
#' - logLik(tm) / nrow(wine)
#'
#' @export
k_ontram <- function(
  mod_baseline,
  list_of_shift_models = NULL,
  complex_intercept = FALSE,
  ...
) {
  if (is.null(list_of_shift_models)) {
    list_of_shift_models <- keras_model_sequential() %>%
      layer_dense(units = 1L, input_shape = c(1L),
                  kernel_initializer = initializer_zeros(),
                  use_bias = FALSE, trainable = FALSE)
  }
  nshift <- length(list_of_shift_models)
  if (nshift == 1L) {
    shift_in <- list_of_shift_models$input
    shift_out <- list_of_shift_models$output
  } else if (nshift >= 2L) {
    shift_in <- lapply(list_of_shift_models, function(x) x$input)
    shift_out <- lapply(list_of_shift_models, function(x) x$output) %>%
      layer_add()
  }
  inputs <- list(mod_baseline$input, shift_in)
  outputs <- list(mod_baseline$output, shift_out)
  m <- keras_model(inputs = inputs, outputs = layer_concatenate(outputs))
  m$mod_baseline <- mod_baseline
  m$list_of_shift_models <- list_of_shift_models
  if (complex_intercept) {
    class(m) <- c("k_ontram_ci", "k_ontram", class(m))
  } else {
    class(m) <- c("k_ontram", class(m))
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
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = TRUE]
    yu <- y_true[, 1L:(K - 1L), drop = FALSE]
    yl <- y_true[, 2L:K, drop = FALSE]
    upr <- k_sum(tf$multiply(yu, intercepts), axis = 0L) - shifts
    lwr <- k_sum(tf$multiply(yl, intercepts), axis = 0L) - shifts
    t1 <- y_true[, 1L, drop = TRUE]
    tK <- y_true[, K, drop = TRUE]
    lik <- t1 * k_sigmoid(upr) + tK * (1 - k_sigmoid(lwr)) +
      (1 - t1) * (1 - tK) * (k_sigmoid(upr) - k_sigmoid(lwr))
    - k_mean(k_log(lik))
  }
}

#' CRPS loss
#' @examples
#' rps_loss <- k_ontram_rps(ncol(Y))
#' @export
k_ontram_rps <- function(K) {
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = FALSE]
    y_cum <- tf$cumsum(y_true, axis = 1L)
    cdf <- k_sigmoid(intercepts - shifts)
    briers <- (cdf - y_cum[, 1L:(K - 1L), drop = FALSE])^2
    k_mean(k_mean(briers, axis = 1L))
  }
}

#' RPS loss
#' @examples
#' rps_loss <- k_ontram_rps(ncol(Y))
#' @export
k_ontram_rps <- function(K) {
  function(y_true, y_pred) {
    intercepts <- y_pred[, 1L:(K - 1L), drop = FALSE]
    shifts <- y_pred[, K, drop = FALSE]
    y_cum <- tf$cumsum(y_true, axis = 1L)
    cdf <- k_sigmoid(intercepts - shifts)
    briers <- (cdf - y_cum[, 1L:(K - 1L), drop = FALSE])^2
    k_mean(k_mean(briers, axis = 1L))
  }
}

#' Layer for transforming raw intercepts using softplus function
#' @export
layer_trafo_intercept <- function() {
  tf$keras$layers$Lambda(
    function(x) {
      w1 <- x[, 1L, drop = FALSE]
      wrest <- tf$math$log(1L + tf$math$exp(x[, 2L:x$shape[[2]], drop = FALSE]))
      tf$cumsum(k_concatenate(list(w1, wrest), axis = 0L), axis = 1L)
    }
  )
}

#' Baseline model
#' @export
k_mod_baseline <- function(K, ...) {
  keras_model_sequential() %>%
    layer_dense(units = K - 1L, input_shape = 1L, use_bias = FALSE,
                ... = ...) %>%
    layer_trafo_intercept()()
}

#' gamma to theta
#' @examples
#' .to_theta(c(-1, 1, 1))
.to_theta <- function(gammas) {
  return(c(gammas[1], gammas[1] + cumsum(log(1 + exp(gammas[-1])))))
}

#' theta to gamma
#' @examples
#' .to_gamma(.to_theta(c(-1, 1, 1)))
.to_gamma <- function(thetas) {
  gammas <- c(thetas[1L], log(exp(diff(thetas)) - 1))
  if(any(!is.finite(gammas))) {
    gammas[!is.finite(gammas)] <- 1e-20
  }
  return(gammas)
}

#' S3 methods for \code{k_ontram}
#' @method predict k_ontram
#' @export
predict.k_ontram <- function(object, x,
                             type = c("distribution", "density", "trafo",
                                      "baseline_only", "hazard", "cumhazard",
                                      "survivor", "odds", "raw"),
                             ...) {
  type <- match.arg(type)
  if ("k_ontram_ci" %in% class(object)) {
    class(object) <- class(object)[-2L]
  } else {
    class(object) <- class(object)[-1L]
  }
  preds <- predict(object, x = x, ... = ...)
  if (type == "raw")
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

#' Function for estimating a \code{k_ontram} or \code{k_ontram_ci} model
#' @examples
#' set.seed(2021)
#' mbl_si <- k_mod_baseline(5L)
#' msh <- mod_shift(2L)
#' mbl_ci <- keras_model_sequential() %>%
#'    layer_dense(units = 8, input_shape = 1L, activation = "relu") %>%
#'    layer_dense(units = 5L, use_bias = FALSE, activation = "linear") %>%
#'    layer_trafo_intercept()()
#' m1 <- k_ontram(mbl_si, msh)
#' m2 <- k_ontram(mbl_ci, complex_intercept = TRUE)
#'
#' data("wine", package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' x_train <- ontram:::.rm_int(model.matrix(~ temp + contact, data = wine[20:nrow(wine), ]))
#' im_train <- ontram:::.rm_int(model.matrix(~ noise, data = wine[20:nrow(wine), ]))
#' x_val <- ontram:::.rm_int(model.matrix(~ temp + contact, data = wine[1:19, ]))
#' im_val <- ontram:::.rm_int(model.matrix(~ noise, data = wine[1:19, ]))
#' y_train <- model.matrix(~ 0 + rating, data = wine[20:nrow(wine), ])
#' y_val <- model.matrix(~ 0 + rating, data = wine[1:19, ])
#'
#' loss <- k_ontram_loss(ncol(y_train))
#' compile(m1, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
#' fit_k_ontram(m1, x = x_train, y = y_train, batch_size = nrow(wine),
#'              validation_data = list(x_val, y_val), epoch = 10, view_metrics = FALSE)
#' compile(m2, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-4))
#' fit_k_ontram(m2, x = im_train, y = y_train, batch_size = nrow(wine),
#'              validation_data = list(im_val, y_val), epoch = 10, view_metrics = FALSE)
#' @export
fit_k_ontram <- function(object, x, validation_data = NULL, ...) {
  if (!("k_ontram_ci" %in% class(object))) {
    if (is.list(x)) {
      x <- c(list(matrix(1, nrow = nrow(x[[1]]))), x)
    } else {
      x <- c(list(matrix(1, nrow = nrow(x))), list(x))
    }
    if (!is.null(validation_data)) {
      if (is.list(validation_data[[1]])) {
        validation_data <- list(c(list(matrix(1, nrow = nrow(validation_data[[2]]))),
                                  validation_data[[1]]),
                                validation_data[[2]])
      } else {
        validation_data <- list(c(list(matrix(1, nrow = nrow(validation_data[[2]]))),
                                  list(validation_data[[1]])),
                                validation_data[[2]])
      }
    }
  } else if ("k_ontram_ci" %in% class(object)) {
    if (!(is.list(x))) {
      x <- list(x, matrix(1, nrow = nrow(x)))
    }
    if (!is.null(validation_data)) {
      validation_data <- list(c(list(validation_data[[1]]),
                                list(matrix(1, nrow = nrow(validation_data[[2]])))),
                              validation_data[[2]])
    }
  }
  fit(object, x = x, validation_data = validation_data, ...)
}

#' Function for estimating a \code{k_ontram} or \code{k_ontram_ci} model with augmented images as input
#' @examples
#' mnist <- dataset_mnist()
#' c(c(x_train, y_train), c(x_val, y_val)) %<-% mnist
#' y_train <- to_categorical(y_train)
#' y_val <- to_categorical(y_val)
#' x_train <- array_reshape(x_train, c(60000, 28, 28, 1))
#' x_val <- array_reshape(x_val, c(10000, 28, 28, 1))
#' x_train <- x_train / 255
#' x_val <- x_val / 255
#' nim_train <- 100
#' nim_val <- 50
#' x_train <- x_train[1:nim_train, , , , drop = FALSE]
#' y_train <- y_train[1:nim_train, , drop = FALSE]
#' x_val <- x_val[1:nim_val, , , , drop = FALSE]
#' y_val <- y_val[1:nim_val, , drop = FALSE]
#'
#' mbl <- k_mod_baseline(ncol(y_train))
#' mim <-  keras_model_sequential() %>%
#'   layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
#'                 input_shape = c(28, 28, 1)) %>%
#'   layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#'   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
#'   layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#'   layer_flatten() %>%
#'   layer_dense(units = 32, activation = "relu") %>%
#'   layer_dense(units = 1L)
#' m <- k_ontram(mbl, mim)
#' compile(m, optimizer = optimizer_adam(learning_rate = 10^-4), loss = k_ontram_loss(ncol(y_train)))
#'
#' datagen <- image_data_generator(
#'   rotation_range = 20,
#'   width_shift_range = 0.2,
#'   height_shift_range = 0.2,
#'   shear_range = 0.15,
#'   zoom_range = 0.15,
#'   fill_mode = "nearest"
#' )
#'
#' f <- fit_k_ontram_augmented_data(m = m, im_train = x_train, im_val = x_val, y_train = y_train, y_val = y_val,
#'                                     generator = datagen, epochs = 10, mim_as_mbl = FALSE, bs = 32,
#'                                     history = TRUE, save_best_only = TRUE)
#' f$hist
#' f$best_epoch
#' @export
fit_k_ontram_augmented_data <- function(m, im_train, im_val = NULL, x_train = NULL, x_val = NULL, y_train, y_val = NULL,
                                        generator, epochs = 10, mim_as_mbl = FALSE, bs = 32,
                                        history = TRUE, save_best_only = TRUE, patience = 1, filepath = NULL) {
  ndim <- length(dim(im_train))
  if (ndim == 5) {
    gen <- flow_images_from_data(im_train[,,,,], generator = generator,
                                 batch_size = dim(im_train)[1], shuffle = FALSE)
  } else if(ndim == 4) {
    gen <- flow_images_from_data(im_train, generator = generator,
                                 batch_size = dim(im_train)[1], shuffle = FALSE)
  }
  m_hist <- list()
  train_loss <- numeric(epochs)
  val_loss <- numeric(epochs)
  for (epo in seq_len(epochs)) {
    cat("Epoch", epo, "\n")
    im_train_aug <- array(generator_next(gen), dim = dim(im_train))
    inp_val <- NULL
    if (!is.null(x_train)) {
      if (mim_as_mbl) {
        inp_train <- list(im_train_aug, x_train)
        if (!is.null(x_val)) {
          inp_val <- list(list(im_val, x_val), y_val)
        }
      } else {
        inp_train <- list(matrix(1, nrow = nrow(im_train)), x_train, im_train_aug)
        if (!is.null(x_val)) {
          inp_val <- list(list(matrix(1, nrow = nrow(im_val)), x_val, im_val), y_val)
        }
      }
    } else if(is.null(x_train)) {
      if (mim_as_mbl) {
        inp_train <- list(im_train_aug, matrix(1, nrow = nrow(im_train)))
        if (!is.null(im_val)) {
          inp_val <- list(list(im_val, matrix(1, nrow = nrow(im_val))), y_val)
        }
      } else {
        inp_train <- list(matrix(1, nrow = nrow(im_train)), im_train_aug)
        if (!is.null(im_val)) {
          inp_val <- list(list(matrix(1, nrow = nrow(im_val)), im_val), y_val)
        }
      }
    }
    m_hist[[epo]] <- fit(m, x = inp_train, y = y_train, batch_size = bs,
                         epochs = 1L, validation_data = inp_val, shuffle = TRUE)
    train_loss[epo] <- m_hist[[epo]]$metrics$loss
    val_loss[epo] <- m_hist[[epo]]$metrics$val_loss
    if (save_best_only) {
      current_val_loss <- m_hist[[epo]]$metrics$val_loss
      if (epo == 1) {
        final_weights <- best_weights <- get_weights(m)
        final_epoch <- current_best_epoch <- epo
        best_val_loss <- m_hist[[epo]]$metrics$val_loss
        n_worse <- 0
      }
      if (!is.na(current_val_loss <= best_val_loss) &
                 current_val_loss <= best_val_loss) {
        best_weights <- get_weights(m)
        best_val_loss <- m_hist[[epo]]$metrics$val_loss
        current_best_epoch <- epo
        n_worse <- 0
      }
      if (is.na(current_val_loss > best_val_loss) |
                 current_val_loss > best_val_loss | ((epo == epochs) & (current_val_loss == best_val_loss))) {
        n_worse <- n_worse + 1
        if (n_worse == patience) {
          final_weights <- best_weights
          final_epoch <- current_best_epoch
          n_worse <- 0
        }
      }
    }
  }
  hist_ret <- list(params = list(verbose = m_hist[[1]]$params$verbose,
                                 epochs = epochs,
                                 steps = m_hist[[1]]$params$steps),
                   metrics = list(loss = train_loss,
                                  val_loss = val_loss))
  set_weights(m, final_weights)
  if (!is.null(filepath)) {
    save_model_weights_hdf5(m, filepath = filepath)
  }
  if (history) {
    return(list(hist = hist_ret, best_epoch = final_epoch))
  }
  return(invisible(m))
}

#' Save history of keras model
#' @export
save_k_hist <- function(object, filepath) {
  df <- data.frame(loss = unlist(object$metrics),
                   Type = factor(c(rep("Train", object$params$epochs),
                                   rep("Val", object$params$epochs))),
                   epoch = seq_len(object$params$epochs))

  write.csv(df, file = filepath, row.names = FALSE)
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


