
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
