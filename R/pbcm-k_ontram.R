#' Parametric Bootstrap Cross-fitting Method
#' @examples
#' set.seed(2021)
#' library(ggplot2)
#' data(wine, package = "ordinal")
#' wine$noise <- rnorm(nrow(wine))
#' y <- model.matrix(~ 0 + rating, data = wine)
#' x <- ontram:::.rm_int(model.matrix(rating ~ temp + contact, data = wine))
#' im <- ontram:::.rm_int(model.matrix(rating ~ noise, data = wine))
#' int <- matrix(1, nrow = nrow(wine))
#' loss <- k_ontram_loss(ncol(y))
#' mbl <- k_mod_baseline(ncol(y))
#' msh <- mod_shift(ncol(x))
#' mim <- keras_model_sequential(name = "complex_shift") %>%
#'    layer_dense(input_shape = 1L, units = 8, activation = "relu") %>%
#'    layer_dense(units = 1, use_bias = FALSE, activation = "linear")
#' mo1 <- k_ontram(mbl, msh)
#' mo2 <- k_ontram(mbl, list(msh, mim))
#' m <- list(mo1, mo2)
#' x <- list(list(int, x), list(int, x, im))
#' pbcm_out <- pbcm(m, x, y, n_split = 3)
#' plot(pbcm_out)
#' plot(pbcm_out, obs_av = TRUE)
#' @param m list of models (either of class \code{\link{k_ontram}} or \code{\link[tram]{tram}}).
#' @param x list containing data for each model separately (as list).
#' @param y response variable (one-hot encoded).
#' @param n_split number of random splits.
#' @param n_npBoot number of non-parametric bootstrap samples (after each random split).
#' @param n_sim number of simulated response variables.
#' @param split proportion of data used for training, testing and validating the models. The proportion for validating the models is calculated with respect to the training set (e.g. 10\% of 90\% of all data).
#' @param epochs vector containing number of epochs for each model.
#' @param loss loss function.
#' @param filepath path where model checkpoints are saved (get removed after training).
#' @param callback_model_ckpts_args list of arguments used as input for function \code{\link[keras]{callback_model_checkpoint}}.
#' @param img_augmentation logical. Whether to augment training data.
#' @param aug_params list of arguments used as input for function \code{\link[keras]{image_data_generator}}.
#' @export
pbcm <- function(m, x, y, n_split = 5, n_npBoot = 1, n_sim = 1, split = c(0.9, 0.1, 0.1),
                 optimizer = optimizer_adam(learning_rate = 1e-2), batch_size = 7,
                 epochs = rep(10, length(m)), loss = k_ontram_loss(ncol(y)),
                 filepath = "weights_best.h5",
                 callback_model_ckpts_args = list(monitor = "val_loss", save_best_only = TRUE,
                                                  save_weights_only = TRUE, mode = "min"),
                 img_augmentation = FALSE, aug_params = list(horizontal_flip = FALSE,
                                                             vertical_flip = FALSE,
                                                             zoom_range = 0.1,
                                                             rotation_range = 5,
                                                             shear_range = 0,
                                                             width_shift_range = 0.1,
                                                             height_shift_range = 0.1,
                                                             fill_mode = "nearest"),
                 seed = NULL) {
  stopifnot(split[1L] + split[2L] == 1L)
  if (!(is.null(seed))) {
    set.seed(seed)
  }
  n_mod <- length(m)
  K <- ncol(y)
  nll_sim <- array(numeric(0), dim = c(n_split * n_npBoot, n_mod, n_mod))
  dimnames(nll_sim) <- list(NULL, NULL, paste0(rep("sampled_mod_", n_mod), 1L:n_mod))
  nll_obs <- vector(mode = "list", n_mod)

  ## RANDOM SPLITT ##
  for (s in seq_len(n_split)) {
    lys_split <- .rand_split(x, y, split)
    x_train <- lys_split$x_train
    x_test <- lys_split$x_test
    y_train <- lys_split$y_train
    y_test <- lys_split$y_test

    ## NON-PARA BOOTSTRAP ##
    for (b in seq_len(n_npBoot)) {
      lys_npBoot <- .npBoot(x_train, x_test, y_train, y_test)
      x_train <- lys_npBoot$x_train
      x_test <- lys_npBoot$x_test
      y_train <- lys_npBoot$y_train
      y_test <- lys_npBoot$y_test

      ## FIT MODELS ##
      m_fit <- mapply(function(mod, x_train, epochs, n_mod) {
        if ("k_ontram" %in% class(mod)) {
          filepath <- paste0("ckpts/best_weights_m", n_mod)
          callback_args <- c(filepath, callback_model_ckpts_args)
          callbacks <- list(do.call(callback_model_checkpoint, callback_args))
          compile(mod, optimizer = optimizer, loss = loss)
          fit(mod, x = x_train, y = y_train, batch_size = batch_size, epochs = epochs,
              validation_split = split[3L], shuffle = TRUE, callbacks = callbacks,
              view_metrics = FALSE)
          load_model_weights_hdf5(mod, filepath)
        } else if ("Polr" %in% class(mod)) {
          y_name <- names(mod$data)[1L]
          fm <- deparse(formula(getCall(mod)))
          fm <- formula(sub(y_name, "y_train", fm))
          y_train <- apply(y_train, 1, which.max)
          x_train <- x_train[[1]]
          df_train <- data.frame(cbind(y_train, x_train))
          df_train$y_train <- ordered(df_train$y_train, levels = 1:K)
          names(df_train)[-1L] <- names(mod$data)[-1L]
          mod <- Polr(fm, data = df_train)
        }
        return(invisible(mod))
      }, mod = m, x_train = x_train, epochs = epochs, n_mod = 1:n_mod, SIMPLIFY = FALSE)
      unlink("ckpts", recursive = TRUE)

      ## CALC OBSERVED NLL ##
      nll <- mapply(function(mod, x_test) {
        if ("k_ontram" %in% class(mod)) {
          as.numeric(loss(k_constant(y_test), mod(x_test)))
        } else if ("Polr" %in% class(mod)) {
          y_test <- apply(y_test, 1, which.max)
          x_test <- x_test[[1]]
          df_test <- data.frame(cbind(y_test, x_test))
          df_test$y_test <- ordered(df_test$y_test, levels = 1:K)
          names(df_test) <- names(mod$data)
          -logLik(mod, newdata = df_test)/nrow(df_test)
        }
      }, mod = m_fit, x_test = x_test, SIMPLIFY = FALSE)
      nll_obs <- mapply(function(nll_obs, nll) {
        append(nll_obs, nll)
      }, nll_obs = nll_obs, nll = nll, SIMPLIFY = FALSE)

      ## SIMULATE NEW RESPONSES ##
      y_sim_train <- mapply(function(mod, x_train) {
        if ("k_ontram" %in% class(mod)) {
          ret <- simulate(mod, x = x_train, nsim = n_sim)
          ret <- model.matrix(~ 0 + unlist(ret))
        } else if ("Polr" %in% class(mod)) {
          df_train <- data.frame(x_train[[1L]])
          names(df_train) <- names(mod$data)[-1L]
          ret <- simulate(mod, newdata = df_train, nsim = n_sim)
          ret <- model.matrix(~ 0 + unlist(ret))
        }
        return(ret)
      }, mod = m_fit, x_train = x_train, SIMPLIFY = FALSE)
      x_train <- lapply(x_train, function(x_train) {
        lapply(x_train, function(x) {
          n_dim <- length(dim(x))
          if (n_dim == 2L) {
            do.call("rbind", rep(list(x), n_sim))
          } else {
            dim_new <- dim(x)
            dim_new[3L] <- n_sim * dim_new[3L]
            array(replicate(n_sim, x), dim = dim_new)
          }
        })
      })
      y_sim_test <- mapply(function(mod, x_test) {
        if ("k_ontram" %in% class(mod)) {
          ret <- simulate(mod, x = x_test, nsim = n_sim)
          ret <- model.matrix(~ 0 + unlist(ret))
        } else if ("Polr" %in% class(mod)) {
          df_test <- data.frame(x_test[[1L]])
          names(df_test) <- names(mod$data)[-1L]
          ret <- simulate(mod, newdata = df_test, nsim = n_sim)
          ret <- model.matrix(~ 0 + unlist(ret))
        }
        return(ret)
      }, mod = m_fit, x_test = x_test, SIMPLIFY = FALSE)
      x_test <- lapply(x_test, function(x_test) {
        lapply(x_test, function(x) {
          n_dim <- length(dim(x))
          if (n_dim == 2L) {
            do.call("rbind", rep(list(x), n_sim))
          } else {
            dim_new <- dim(x)
            dim_new[3L] <- n_sim * dim_new[3L]
            array(replicate(n_sim, x), dim = dim_new)
          }
        })
      })

      ## REFIT MODELS ##
      m_refit <- mapply(function(mod, x_train, epochs, n_mod) {
        mapply(function(y_sim_train, n_fit) {
          if ("k_ontram" %in% class(mod)) {
            .reset_weights(mod)
            mbl <- mod$mod_baseline
            msh <- mod$list_of_shift_models
            if (length(msh) == 1L) {
              mbl <- clone_model(mbl)
              msh <- clone_model(msh)
              mod_new <- k_ontram(mbl, msh)
            } else if (length(msh) >= 2L) {
              mbl <- clone_model(mbl)
              tmp <- vector(mode = "list", length(msh))
              for (idx in seq_along(msh)) {
                tmp[[idx]] <- clone_model(msh[[idx-1L]])
              }
              msh <- tmp
            }
            mod_new <- k_ontram(mbl, msh)
            filepath <- paste0("ckpts/best_weights_m", n_mod, "_refit", n_fit)
            callback_args <- c(filepath, callback_model_ckpts_args)
            callbacks <- list(do.call(callback_model_checkpoint, callback_args))
            compile(mod_new, optimizer = optimizer, loss = loss)
            fit(mod_new, x = x_train, y = y_sim_train, batch_size = batch_size, epochs = epochs,
                validation_split = split[3L], shuffle = TRUE, callbacks = callbacks,
                view_metrics = FALSE)
            load_model_weights_hdf5(mod_new, filepath)
          } else if ("Polr" %in% class(mod)) {
            y_name <- names(mod$data)[1L]
            fm <- deparse(formula(getCall(mod)))
            fm <- formula(sub(y_name, "y_sim_train", fm))
            y_sim_train <- apply(y_sim_train, 1, which.max)
            df_train <- data.frame(cbind(y_sim_train, unlist(x_train[[1L]])))
            df_train$y_sim_train <- ordered(df_train$y_sim_train, levels = 1:K)
            names(df_train)[-1L] <- names(mod$data)[-1L]
            mod_new <- Polr(fm, data = df_train)
          }
          return(mod_new)
        }, y_sim_train = y_sim_train, n_fit = 1L:length(m), SIMPLIFY = FALSE)
      }, mod = m, x_train = x_train, epochs = epochs, n_mod = 1L:n_mod, SIMPLIFY = FALSE)
      unlink("ckpts", recursive = TRUE)
      m_refit <- unlist(m_refit, recursive = FALSE)

      ## CALC NLL ##
      x_test <- rep(x_test, each = n_mod)
      nll <- mapply(function(mod, x_test) {
        lapply(y_sim_test, function(y_sim_test) {
          if ("k_ontram" %in% class(mod)) {
            as.numeric(loss(k_constant(y_sim_test), mod(x_test)))
          } else if ("Polr" %in% class(mod)) {
            y_test <- apply(y_sim_test, 1, which.max)
            x_test <- x_test[[1]]
            df_test <- data.frame(cbind(y_test, x_test))
            df_test$y_test <- ordered(df_test$y_test, levels = 1:K)
            names(df_test) <- names(mod$data)
            -logLik(mod, newdata = df_test)/nrow(df_test)
          }
        })
      }, mod = m_refit, x_test = x_test, SIMPLIFY = FALSE)
      for (fit in seq_len(n_mod)) {
        for (sim in seq_len(n_mod)) {
          if (s == 1L) {
            nll_sim[b, fit, sim] <- nll[[fit]][[sim]]
          } else if (s >= 2L) {
            nll_sim[s-1*n_npBoot+b, fit, sim] <- nll[[fit]][[sim]]
          }
        }
      }
    }
  }
  ret <- list(nll_sim = nll_sim,
              nll_obs = nll_obs)
  class(ret) <- "pbcm"
  return(ret)
}

#' Plot PBCM
#' @param object output of function \code{\link{pbcm}}.
#' @param obs_av logical. Whether to plot mean of observed NLL as line (instead of ECDF).
#' @param titles titles used for facets.
#' @param labels labels used for fitted models.
#' @param cols colors.
#' @param n_col,n_row number of columns and rows for displaying plots.
#' @export
plot.pbcm <- function(object, obs_av = FALSE,
                      titles = paste0("Sampled~from~Model~", 1:dim(object[[1]])[3]),
                      labels = paste0("Mod ", 1:dim(object[[1]])[3]),
                      cols = 1:dim(object[[1]])[3],
                      n_col = NULL, n_row = NULL) {
  nll_sim <- object$nll_sim
  nll_obs <- object$nll_obs
  n_mod <- dim(nll_sim)[3]
  n_fit <- nrow(nll_sim)
  min_x <- min(unlist(c(nll_sim, nll_obs)))
  max_x <- max(unlist(c(nll_sim, nll_obs)))
  if (obs_av) {
    nll_obs <- lapply(nll_obs, mean)
  }
  nll <- lapply(seq_len(n_mod), function(idx) {
    nll_sim <- as.data.frame(nll_sim[,,idx])
    nll_sim$sim <- factor(idx)
    return(nll_sim)
  })
  nll <- do.call("rbind", nll)
  nll <- reshape(nll, direction = "long",
                 varying = list(seq_len(n_mod)),
                 v.names = "nll_sim",
                 timevar = "fit")
  nll$fit <- as.factor(nll$fit)
  nll <- nll[order(nll$sim), ]
  if (!is.null(titles)) {
    levels(nll$sim) <- titles
  }
  if (any(sapply(nll_obs, length) >= 2)) {
    nll$nll_obs <- rep(do.call("c", nll_obs), n_mod)
  }
  plt <- ggplot(nll) +
    stat_ecdf(aes(nll_sim, colour = fit), size = 1) +
    facet_wrap(~ sim, ncol = n_col, nrow = n_row,
               labeller = label_parsed, scales = "free") +
    xlim(c(min_x - 0.001, max_x + 0.001)) +
    theme_bw() +
    labs(x = "Test NLL", y = "ECDF") +
    scale_color_manual(name = "Fitted",
                       labels = labels, values = cols)
  if (any(sapply(nll_obs, length) == 1)) {
    nll_obs <- data.frame(xint = do.call("rbind", nll_obs))
    nll_obs$cols <- factor(cols)
    plt <- plt + geom_vline(data = nll_obs, aes(xintercept = xint, color = cols),
                            linetype = "dashed", size = 0.5, show.legend = FALSE)
  } else if (any(sapply(nll_obs, length) >= 2)) {
    plt <- plt + stat_ecdf(aes(nll_obs, colour = fit), linetype = "dashed", size = 0.5)
  }
  return(plt)
}

# Helper functions
.rand_split <- function(x, y, split) {
  idx <- 1:dim(x[[1]][[1]])[1]
  idx_train <- sample(idx, ceiling(split[1] * max(idx)), replace = FALSE)
  idx_test <- idx[-idx_train]
  n_dim_y <- length(dim(y))
  x_train <- lapply(x, function(x) {
    lapply(x, function(x) {
      n_dim <- length(dim(x))
      eval(parse(text = paste0("x[idx_train", paste0(rep(",", n_dim), collapse = ""), "drop = FALSE]")))
    })
  })
  x_test <- lapply(x, function(x) {
    lapply(x, function(x) {
      n_dim <- length(dim(x))
      eval(parse(text = paste0("x[idx_test", paste0(rep(",", n_dim), collapse = ""), "drop = FALSE]")))
    })
  })
  y_train <- eval(parse(text = paste0("y[idx_train", paste0(rep(",", n_dim_y), collapse = ""), "drop = FALSE]")))
  y_test <- eval(parse(text = paste0("y[idx_test", paste0(rep(",", n_dim_y), collapse = ""), "drop = FALSE]")))
  ret <- list(x_train = x_train,
              x_test = x_test,
              y_train = y_train,
              y_test = y_test)
  return(ret)
}

.npBoot <- function(x_train, x_test, y_train, y_test) {
  n_train <- dim(x_train[[1]][[1]])[1]
  n_test <- dim(x_test[[1]][[1]])[1]
  idx_train <- sample(1:n_train, n_train, replace = TRUE)
  idx_test <- sample(1:n_test, n_test, replace = TRUE)
  n_dim_y <- length(dim(y_train))
  x_train <- lapply(x_train, function(x) {
    lapply(x, function(x) {
      n_dim <- length(dim(x))
      eval(parse(text = paste0("x[idx_train", paste0(rep(",", n_dim), collapse = ""), "drop = FALSE]")))
    })
  })
  x_test <- lapply(x, function(x) {
    lapply(x, function(x) {
      n_dim <- length(dim(x))
      eval(parse(text = paste0("x[idx_test", paste0(rep(",", n_dim), collapse = ""), "drop = FALSE]")))
    })
  })
  y_train <- eval(parse(text = paste0("y[idx_train", paste0(rep(",", n_dim_y), collapse = ""), "drop = FALSE]")))
  y_test <- eval(parse(text = paste0("y[idx_test", paste0(rep(",", n_dim_y), collapse = ""), "drop = FALSE]")))
  ret <- list(x_train = x_train,
              x_test = x_test,
              y_train = y_train,
              y_test = y_test)
  return(ret)
}

.reset_weights <- function(x) {
  initializer <- tf$keras$initializers$glorot_uniform()
  w <- get_weights(x)
  for (idx in seq_along(w)) {
    input_shape <- dim(w[[idx]])
    n_dim <- length(input_shape)
    if (n_dim == 1) {
      w_reset <- tf$Variable(initializer(shape = shape(input_shape)),
                             dtype = "float32")
    } else {
      w_reset <- tf$Variable(initializer(shape = shape(input_shape[1], input_shape[2])),
                             dtype = "float32")
    }
    w_reset <- array(as.numeric(w_reset), dim = input_shape)
    w[[idx]] <- w_reset
  }
  set_weights(x, w)
  return(invisible(x))
}
