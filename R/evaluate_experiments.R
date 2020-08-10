#' @export
get_accuracy <- function(target, predicted) {
  ret <- mean(predicted == target)
  return(ret)
}

#' @export
get_confmat <- function(target, predicted) {
  ret <- table(predicted, target)
  return(ret)
}

#' Evaluate fitted ontram models
#' @examples
#' data("wine", package = "ordinal")
#' fml <- rating ~ temp + contact
#' x_train <- model.matrix(fml, data = wine)[, -1L]
#' y_train <- model.matrix(~ 0 + rating, data = wine)
#' x_valid <- x_train[1:20,]
#' y_valid <- y_train[1:20,]
#' mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
#'     method = "logit", n_batches = 10, epochs = 10)
#' mo_hist <- fit_ontram(mo, x_train = x_train, y_train = y_train, history = TRUE,
#'     x_test = x_valid, y_test = y_valid)
#' debugonce(eval_batchwise)
#' out <- eval_batchwise(mo, x_valid, y_valid, NULL, 5, as.numeric(wine$rating))
#' loss_from_eval(out$preds)
#' plots_from_eval(out$preds)
#' @export
eval_batchwise <- function(model, x, y, im, bs, target) {
  n <- nrow(y)
  idxs <- sample(rep(seq_len(bs), ceiling(n/bs)), n)
  preds <- list()
  confmat <- list()
  acc <- c()
  for (bat in seq_len(bs)) {
    idx <- which(idxs == bat)
    yb <- .batch_subset(y, idx, dim(y))
    xb <- if (is.null(x)) {
      NULL
    } else {
      .batch_subset(x, idx, dim(x))
    }
    imb <- if (is.null(im)) {
      NULL
    } else {
      .batch_subset(im, idx, dim(im))
    }
    tb <- target[idx]
    preds[[bat]] <- predict(model, xb, yb, imb)
    acc <- append(acc, get_accuracy(tb, preds[[bat]]$response))
    confmat[[bat]] <- get_confmat(tb, preds[[bat]]$response)
  }
  ret <- list(preds = preds, accuracy = acc, confusion_mat = confmat)
  return(ret)
}

#' @export
loss_from_eval <- function(preds) {
  mean(do.call("c", lapply(preds, function(x) x$negLogLik)))
}

#' @export
plots_from_eval <- function(preds) {
  lapply(preds, function(x) matplot(t(x$pdf), type = "l"))
  return(invisible(NULL))
}
