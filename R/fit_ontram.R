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
fit_ontram <- function(model, history = FALSE, x_train = NULL,
                      y_train, img_train = NULL, save_model = FALSE,
                      x_test = NULL, y_test = NULL, img_test = NULL) {
  stopifnot(nrow(x_train) == nrow(y_train))
  stopifnot(ncol(y_train) == model$y_dim)
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
    class(model_history) <- "ontram_history"
    hist_idxs <- sample(rep(seq_len(10), ceiling(n/10)), n)
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
    }
  }
  end_time <- Sys.time()
  message(paste0("Training took ", end_time - start_time))
  if (history)
    return(model_history)
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
                       x_test = NULL, y_test = NULL, img_test = NULL) {
  stopifnot(nrow(x_train) == nrow(y_train))
  stopifnot(ncol(y_train) == model$y_dim)
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
