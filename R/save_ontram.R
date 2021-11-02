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
  nm_eta <- paste0(filename, "_eta.h5")
  nm_rest <- paste0(filename, "_r.Rds")
  load(nm_rest)
  mt <- load_model_hdf5(nm_theta)
  if (file.exists(nm_beta)) { 
    mb <- load_model_hdf5(nm_beta)
  } else {
    mb <- NULL
  }
  if (file.exists(nm_eta)) {
    me <- load_model_hdf5(nm_eta)
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
    history$epoch_best <- df[3, 1]
  }
  history$train_loss <- as.numeric(df[1, ])
  history$test_loss <- as.numeric(df[2, ])
  class(history) <- "ontram_history"
  return(history)
}

