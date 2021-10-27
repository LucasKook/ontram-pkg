#' Function for estimating the model
#' @examples
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' x_valid <- x_train[1:20,]
#' y_valid <- y_train[1:20,]
#' mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
#'     method = "logit", n_batches = 10, epochs = 50)
#' mo_hist <- fit_ontram(mo, x_train = x_train, y_train = y_train, history = TRUE,
#'     x_test = x_valid, y_test = y_valid)
#' plot(mo_hist)
#' @export
#' @param model an object of class "\link[ontram]{ontram}".
#' @param history logical. If TRUE train and test are returned as list.
#' @param x_train tabular data used for training the model.
#' @param y_train response data (one-hot encoded) used for training the model.
#' @param img_train image data used for training the model.
#' @param x_test tabular data used for evaluating the model.
#' @param y_test response data (one-hot encoded) used for evaluating the model.
#' @param img_test image data used for evaluating the model.
#' @param early_stopping logical. Whether to use early stopping (requires \code{history = TRUE}).
#' @param patience number of epochs with no improvement after which training will be stopped.
#' @param min_delta minimum increase in test loss considered as no improvement.
#' @param stop_train logical. Whether to stop training if conditions are fulfilled for the first time or whether to continue training.
#' @param save_best logical. Whether best model should be saved as HDF file.
#' @param filepath path where to save best model if \code{save_best = TRUE}.
fit_ontram <- function(model, history = FALSE, x_train = NULL,
                       y_train, img_train = NULL,
                       x_test = NULL, y_test = NULL, img_test = NULL,
                       early_stopping = FALSE, patience = 1,
                       min_delta = 0, stop_train = TRUE,
                       filepath = NULL) {
  stopifnot(nrow(x_train) == nrow(y_train))
  stopifnot(ncol(y_train) == model$y_dim)
  if (early_stopping) {
    stopifnot(patiece >= 1)
    stopifnot(min_delta >= 0)
    stopifnot(step_size >= 1)
  }
  apply_gradient_tf <- tf_function(apply_gradient)

  n <- nrow(y_train)
  start_time <- Sys.time()
  message("Training ordinal transformation model neural network.")
  message(paste0("Training samples: ", nrow(y_train)))
  message(paste0("Batches: ", bs <- model$n_batches))
  message(paste0("Batch size: ", ceiling(n/bs)))
  message(paste0("Epochs: ", epo <- model$epoch))
  if (history) {
    model_history <- list(train_loss = c(), test_loss = c())
    if (save_best_only) {
      model_history <- c(model_history, list(epoch_stop = c())) #ag: if save_best_only = T, epoch at which model stopped is saved
    }
    class(model_history) <- "ontram_history"
    hist_idxs <- sample(rep(seq_len(10), ceiling(n/10)), n)
  }
  early_stop <- FALSE #ag: early stopping initials
  if (early_stopping) {
    current_min <- Inf
  }
  for (epo in seq_len(epo)) {
    message(paste0("Training epoch: ", epo))
    batch_idx <- sample(rep(seq_len(bs), ceiling(n/bs)), n)
    for (bat in seq_len(bs)) {
      idx <- which(batch_idx == bat)
      y_batch <- tf$constant(.batch_subset(y_train, idx, dim(y_train)),
                             dtype = "float32")
      if (!is.null(x_train)) {
        x_batch <- tf$constant(.batch_subset(x_train, idx, dim(x_train)),
                               dtype = "float32")
      } else {
        x_batch <- NULL
      }
      if (!is.null(img_train)) {
        img_batch <- tf$constant(.batch_subset(img_train, idx, dim(img_train)),
                                 dtype = "float32")
      } else {
        img_batch <- NULL
      }
      apply_gradient_tf(x_batch, y_batch, model, img_batch,
                        response_varying = model$response_varying)
    }
    if (history) {
      tmp_pred <- c()
      for (hist_bat in seq_len(10)) {
        hist_idx <- which(hist_idxs == hist_bat)
        y_hist <- tf$constant(.batch_subset(y_train, hist_idx, dim(y_train)),
                              dtype = "float32")
        if (!is.null(x_train)) {
          x_hist <- tf$constant(.batch_subset(x_train, hist_idx, dim(x_train)),
                                dtype = "float32")
        } else {
          x_hist <- NULL
        }
        if (!is.null(img_train)) {
          img_hist <- tf$constant(.batch_subset(img_train, hist_idx, dim(img_train)),
                                  dtype = "float32")
        } else {
          img_hist <- NULL
        }
        tmp_pred <- append(tmp_pred,
                           predict(model, x = x_hist,
                                   y = y_hist, im = img_hist)$negLogLik)
      }
      train_loss <- mean(tmp_pred)
      test_loss <- predict(model, x = x_test, y = y_test, im = img_test)$negLogLik
      model_history$train_loss <- append(model_history$train_loss, train_loss)
      model_history$test_loss <- append(model_history$test_loss, test_loss)

      if (early_stopping) { #ag: early stopping implementation
          if (model_history$test_loss[epo] <= current_min) {
            current_min <- model_history$test_loss[epo]
            n_worse <- 0
            if (save_best) {
            save_model.ontram(model, filename = paste0(filepath, "best_model"))
            } else {
              m_best <- model
            }
          } else {
            if (model_history$test_loss[epo] - current_min >= min_delta) {
              n_worse <- n_worse + 1
              if (n_worse == patience) {
                early_stop <- TRUE
                model_history$epoch_stop <- epo - patience
                n_worse <- 0
              }
            }
          }
      }
    }
    if (early_stop) {
      if (save_best) {
        m_best <- load_model.ontram(model, filename = paste0(filepath, "best_model"))
      }
      if (stop_train) {
        message("Early stopping")
        break
      }
    }
  }
  end_time <- Sys.time()
  message(paste0("Training took ", end_time - start_time))
  if (history)
    return(model_history)
  if (early_stopping)
    return(invisible(m_best))
  else
    return(invisible(model))
}


# not very elegant, look for different solution
.batch_subset <- function(obj, idx, dim) {
  ndim <- length(dim)
  if (ndim == 2L) {
    ret <- obj[idx, , drop = FALSE]
  } else if (ndim == 3L) {
    ret <- obj[idx, , , drop = FALSE]
  } else if (ndim == 4L) {
    ret <- obj[idx, , , , drop = FALSE]
  }
  return(ret)
}


#' Function for estimating the model
#' @examples
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' x_valid <- x_train[1:20,]
#' y_valid <- y_train[1:20,]
#' mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
#'     method = "logit", n_batches = 10, epochs = 50)
#' mo_hist <- fit_ontram2(mo, x_train = x_train, y_train = y_train, history = TRUE,
#'     x_test = x_valid, y_test = y_valid)
#' plot(mo_hist)
#' @export
fit_ontram2 <- function(model, history = FALSE, x_train = NULL,
                       y_train, img_train = NULL, save_model = FALSE,
                       x_test = NULL, y_test = NULL, img_test = NULL, verbose = 1) {
  stopifnot(nrow(x_train) == nrow(y_train))
  stopifnot(ncol(y_train) == model$y_dim)
  apply_gradient_tf <- tf_function(apply_gradient)
  n <- nrow(y_train)
  start_time <- Sys.time()
  if (verbose != 0) {
    message("Training ordinal transformation model neural network.")
    message(paste0("Training samples: ", nrow(y_train)))
    message(paste0("Batches: ", bs <- model$n_batches))
    message(paste0("Batch size: ", ceiling(n/bs)))
    message(paste0("Epochs: ", nep <- model$epoch))
    pb <- txtProgressBar(min = 1, max = nep, style = 3)
  }
  if (history) {
    model_history <- list(train_loss = c(), test_loss = c())
    class(model_history) <- "ontram_history"
  }
  for (epo in seq_len(nep)) {
    if (verbose == 2) {
      message(paste0("Training epoch: ", epo))
    } else if (verbose == 1) {
      setTxtProgressBar(pb, epo)
    }
    batch_idx <- sample(rep(seq_len(bs), ceiling(n/bs)), n)
    for (bat in seq_len(bs)) {
      idx <- which(batch_idx == bat)
      y_batch <- tf$constant(.batch_subset(y_train, idx, dim(y_train)),
                             dtype = "float32")
      if (!is.null(x_train)) {
        x_batch <- tf$constant(.batch_subset(x_train, idx, dim(x_train)),
                               dtype = "float32")
      } else {
        x_batch <- NULL
      }
      if (!is.null(img_train)) {
        img_batch <- tf$constant(.batch_subset(img_train, idx, dim(img_train)),
                                 dtype = "float32")
      } else {
        img_batch <- NULL
      }
      apply_gradient_tf(x_batch, y_batch, model, img_batch,
                        response_varying = model$response_varying)
    }
    if (history) {
      train_loss <- predict(model, x = x_train, y = y_train, im = img_train)$negLogLik
      test_loss <- predict(model, x = x_test, y = y_test, im = img_test)$negLogLik
      model_history$train_loss <- append(model_history$train_loss, train_loss)
      model_history$test_loss <- append(model_history$test_loss, test_loss)
    }
  }
  end_time <- Sys.time()
  message(paste0("Training took ", end_time - start_time))
  if (history)
    return(model_history)
  return(invisible(model))
}

#' Function for estimating the model
#' @examples
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' x_valid <- x_train[1:20,]
#' y_valid <- y_train[1:20,]
#' mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
#'     method = "logit", n_batches = 10, epochs = 50)
#' mo_hist <- fit_ontram2(mo, x_train = x_train, y_train = y_train, history = TRUE,
#'     x_test = x_valid, y_test = y_valid)
#' plot(mo_hist)
#' @export
fit_ontram3 <- function(model, history = FALSE, x_train = NULL,
                        y_train, img_train = NULL, save_model = FALSE,
                        x_test = NULL, y_test = NULL, img_test = NULL,
                        lambda2 = 1e-4, numnet = 1) {
  stopifnot(nrow(x_train) == nrow(y_train))
  stopifnot(ncol(y_train) == model$y_dim)
  apply_gradient_tf <- tf_function(apply_gradient2)
  n <- nrow(y_train)
  start_time <- Sys.time()
  message("Training ordinal transformation model neural network.")
  message(paste0("Training samples: ", nrow(y_train)))
  message(paste0("Batches: ", bs <- model$n_batches))
  message(paste0("Batch size: ", ceiling(n/bs)))
  message(paste0("Epochs: ", epo <- model$epoch))
  if (history) {
    model_history <- list(train_loss = c(), test_loss = c())
    class(model_history) <- "ontram_history"
  }
  for (epo in seq_len(epo)) {
    message(paste0("Training epoch: ", epo))
    batch_idx <- sample(rep(seq_len(bs), ceiling(n/bs)), n)
    for (bat in seq_len(bs)) {
      idx <- which(batch_idx == bat)
      y_batch <- tf$constant(.batch_subset(y_train, idx, dim(y_train)),
                             dtype = "float32")
      if (!is.null(x_train)) {
        x_batch <- tf$constant(.batch_subset(x_train, idx, dim(x_train)),
                               dtype = "float32")
      } else {
        x_batch <- NULL
      }
      if (!is.null(img_train)) {
        img_batch <- lapply(img_train, function(x) tf$constant(.batch_subset(x, idx, dim(x)),
                                                             dtype = "float32"))
      } else {
        img_batch <- NULL
      }
      apply_gradient_tf(x_batch, y_batch, model, img_batch,
                        response_varying = model$response_varying,
                        lambda2 = lambda2, numnet = numnet)
    }
    if (history) {
      train_loss <- predict(model, x = x_train, y = y_train, im = img_train)$negLogLik
      test_loss <- predict(model, x = x_test, y = y_test, im = img_test)$negLogLik
      model_history$train_loss <- append(model_history$train_loss, train_loss)
      model_history$test_loss <- append(model_history$test_loss, test_loss)
    }
  }
  end_time <- Sys.time()
  message(paste0("Training took ", end_time - start_time))
  if (history)
    return(model_history)
  return(invisible(model))
}
