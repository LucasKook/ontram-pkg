#' Save ontram model
#' @export
save_model_ontram <- function(object, filename, ...) {
  nm_theta <- paste0(filename, "_theta.h5")
  nm_beta <- paste0(filename, "_beta.h5")
  nm_eta <- paste0(filename, "_eta.h5")
  nm_rest <- paste0(filename, "_r.Rds")
  rest <- list(x_dim = object$x_dim,
               y_dim = object$y_dim,
               n_batches = object$n_batches,
               epochs = object$epochs,
               distr = object$distr,
               lr = object$optimizer$lr$numpy(),
               response_varying = object$response_varying)
  save(rest, file = nm_rest)
  save_model_hdf5(object$mod_baseline, nm_theta)
  if (!is.null(object$mod_shift)) {
    save_model_hdf5(object$mod_shift, nm_beta)
  }
  if (!is.null(object$mod_image)) {
    save_model_hdf5(object$mod_image, nm_eta)
  }
}

#' Load ontram model
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
                           optimizer = tf$keras$optimizers$Adam(learning_rate = rest$lr),
                           distr = rest$distr))
  class(ret) <- "ontram"
  if (ret$response_varying) {
    class(ret) <- c("ontram_rv", class(ret))
  }
  return(ret)
}
