#' Save/Load ontram history
#' @name save_history_ontram
#' @aliases save_ontram_history
#' @aliases load_ontram_history
#' @rdname save_history_ontram
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

#' @rdname save_history_ontram
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
