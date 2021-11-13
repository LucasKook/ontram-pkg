#' Function for estimating the model
#' @examples
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' x_valid <- x_train[1:20,]
#' y_valid <- y_train[1:20,]
#'
#' mo1 <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
#'                    method = "logit", n_batches = 10, epochs = 50)
#' mo1hist <- fit_ontram(mo1, x_train = x_train, y_train = y_train, history = TRUE,
#'                       x_test = x_valid, y_test = y_valid)
#' plot(mo1hist)
#'
#' mbl <- keras_model_sequential() %>%
#'           layer_dense(units = 16, activation = "relu", input_shape = ncol(x_train)) %>%
#'           layer_dense(units = 16, activation = "relu") %>%
#'           layer_dense(units = 16, activation = "relu") %>%
#'           layer_dense(units = 16, activation = "relu") %>%
#'           layer_dense(units = ncol(y_train) - 1L)
#' mo2 <- ontram(mod_bl = mbl, mod_sh = NULL, mod_im = NULL, y_dim = ncol(y_train),
#'               x_dim = NULL, img_dim = NULL, method = "logit",
#'               epochs = 50, response_varying = TRUE)
#' mo2hist <- fit_ontram(mo2, x_train = NULL, y_train = y_train, img_train = x_train,
#'                       x_test = NULL, y_test = y_valid, img_test = x_valid,
#'                       history = TRUE, early_stopping = TRUE, stop_train = FALSE)
#' plot(mo2hist, add_best = TRUE, ylim = c(0, 2.5))
#' @param model an object of class \code{\link{ontram}}.
#' @param history logical. If TRUE train and test loss are returned as a list.
#' @param x_train,y_train,img_train data used for training the model.
#' @param x_test,y_test,img_test data used for evaluating the model.
#' @param verbose logical. Whether to print current training loss when \code{history = TRUE}).
#' @param early_stopping logical. Whether to use early stopping (requires \code{history = TRUE}).
#' @param patience number of epochs with no improvement after which training will be stopped.
#' @param min_delta minimum increase in test loss considered as no improvement.
#' @param stop_train logical. Whether model should be trained for all epochs.
#' @param save_best logical. Whether best model should be saved as HDF file.
#' @param filepath path where to save best model if \code{save_best = TRUE}.
#' @param warm_start logical. Whether initial weights should be non-random.
#' @param weights output output of \code{\link{get_weights_ontram}} or list of similar structure;
#' @param eval_batchwise logical.
#' @param img_augmentation logical. Whether to augment training images.
#' @param aug_params list with arguments used for \code{\link[keras]{image_data_generator}}.
#' @export
fit_ontram <- function(model, history = FALSE, x_train = NULL,
                       y_train, img_train = NULL,
                       x_test = NULL, y_test = NULL, img_test = NULL,
                       verbose = FALSE,
                       early_stopping = FALSE, patience = 1,
                       min_delta = 0, stop_train = FALSE,
                       save_best = FALSE, filepath = NULL,
                       warm_start = FALSE, weights = NULL,
                       eval_batchwise = TRUE,
                       img_augmentation = FALSE,
                       aug_params = list(horizontal_flip = TRUE,
                                         vertical_flip = TRUE,
                                         zoom_range = c(0.7, 1.4),
                                         rotation_range = 30,
                                         width_shift_range = 0.15,
                                         height_shift_range = 0.15,
                                         fill_mode = "nearest")) {
  stopifnot(nrow(x_train) == nrow(y_train))
  stopifnot(ncol(y_train) == model$y_dim)
  if (early_stopping) {
    stopifnot(history)
    stopifnot(patience >= 1)
    stopifnot(min_delta >= 0)
    if (is.null(y_test)) stop("`y_test` not found.")
    if (!is.null(x_train)) {
      if(is.null(x_test)) stop("`x_test` not found.")
    }
    if (!is.null(img_train)) {
      if (is.null(img_test)) stop("`img_test` not found.")
    }
    if (save_best) {
      if (is.null(filepath)) stop("`filepath` is missing.")
      if (!dir.exists(filepath)) dir.create(filepath)
    }
  }
  apply_gradient_tf <- tf_function(apply_gradient)
  n <- nrow(y_train)
  n_test <- nrow(y_test)
  start_time <- Sys.time()
  message("Training ordinal transformation model neural network.")
  message(paste0("Training samples: ", nrow(y_train)))
  message(paste0("Batches: ", bs <- model$n_batches))
  message(paste0("Batch size: ", ceiling(n/bs)))
  message(paste0("Epochs: ", epo <- model$epoch))
  if (history) {
    model_history <- list(train_loss = c(), test_loss = c())
    if (early_stopping) {
      model_history <- c(model_history, list(epoch_best = c()))
    }
    class(model_history) <- "ontram_history"
    hist_idxs <- sample(rep(seq_len(10), ceiling(n/10)), n)
    if (eval_batchwise) {
      hist_idxs_test <- sample(rep(seq_len(bs), ceiling(n_test/bs)), n_test)
    }
  }
  early_stop <- FALSE
  if (early_stopping) {
    current_min <- Inf
  }
  if (warm_start) {
    set_weights_ontram(model, weights = weights)
  }
  for (epo in seq_len(epo)) {
    if (!verbose) {
      message(paste0("Training epoch: ", epo))
    }
    batch_idx <- sample(rep(seq_len(bs), ceiling(n/bs)), n)
    if (img_augmentation) {
      img_3d <- FALSE
      if (length(dim(img_train)) > 4) {
        img_3d <- TRUE
        img_train <- img_train[, , , , ]
      }
      img_generator <- image_data_generator(aug_params)
      img_generator$fit(img_train)
      aug_it <- flow_images_from_data(img_train,
                                      generator = img_generator,
                                      shuffle = FALSE,
                                      batch_size = n)
      img_train <- generator_next(aug_it)
      if (img_3d) {
        img_train <- array_reshape(img_train, c(dim(img_train)[1],
                                              dim(img_train)[2],
                                              dim(img_train)[3],
                                              dim(img_train)[4],
                                              1))
      }
    }
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
      if (eval_batchwise) {
        tmp_pred_test <- c()
        for (hist_bat in seq_len(bs)) {
          hist_idx_test <- which(hist_idxs_test == hist_bat)
          y_hist_test <- tf$constant(.batch_subset(y_test, hist_idx_test, dim(y_test)),
                                     dtype = "float32")
          if (!is.null(x_test)) {
            x_hist_test <- tf$constant(.batch_subset(x_test, hist_idx_test, dim(x_test)),
                                       dtype = "float32")
          } else {
            x_hist_test <- NULL
          }
          if (!is.null(img_test)) {
            img_hist_test <- tf$constant(.batch_subset(img_test, hist_idx_test, dim(img_test)),
                                         dtype = "float32")
          } else {
            img_hist_test <- NULL
          }
          tmp_pred_test <- append(tmp_pred_test,
                                  predict(model, x = x_hist_test,
                                          y = y_hist_test, im = img_hist_test)$negLogLik)
        }
      }
      train_loss <- mean(tmp_pred)
      if (verbose) {
        message(paste0("Training epoch: ", epo, ", Training loss: ", format(round(train_loss, 5), nsmall = 5)))
      }
      if (eval_batchwise) {
        test_loss <- mean(tmp_pred_test)
      } else {
        test_loss <- predict(model, x = x_test, y = y_test, im = img_test)$negLogLik
      }
      model_history$train_loss <- append(model_history$train_loss, train_loss)
      model_history$test_loss <- append(model_history$test_loss, test_loss)
      if (early_stopping) {
          if (model_history$test_loss[epo] <= current_min) {
            current_min <- model_history$test_loss[epo]
            n_worse <- 0
            if (epo == model$epoch) {
              model_history$epoch_best <- epo
            }
            if (save_best) {
            save_model_ontram(model, filename = paste0(filepath, "best_model.h5"))
            } else {
              m_best <- model
            }
          } else {
            if (model_history$test_loss[epo] - current_min >= min_delta) {
              n_worse <- n_worse + 1
              if (n_worse == patience) {
                early_stop <- TRUE
                model_history$epoch_best <- epo - patience
              }
            }
          }
      }
    }
    if (early_stop) {
      if (save_best) {
        m_best <- load_model_ontram(filename = paste0(filepath, "best_model.h5"))
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
  } else if (ndim == 5L) {
    ret <- obj[idx, , , , , drop = FALSE] #ag: added
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
