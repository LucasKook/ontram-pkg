% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/k-ontram.R
\name{k_ontram}
\alias{k_ontram}
\title{Keras interface to ONTRAMs}
\usage{
k_ontram(
  mod_baseline,
  list_of_shift_models = NULL,
  complex_intercept = FALSE,
  ...
)
}
\description{
Keras interface to ONTRAMs
}
\examples{
library(tram)
set.seed(2021)
mbl <- k_mod_baseline(5L, name = "baseline")
msh <- mod_shift(2L, name = "linear_shift")
mim <- mod_shift(1L, name = "complex_shift")
m <- k_ontram(mbl, list(msh, mim))

data("wine", package = "ordinal")
wine$noise <- rnorm(nrow(wine))
X <- .rm_int(model.matrix(~ temp + contact, data = wine))
Y <- model.matrix(~ 0 + rating, data = wine)
Z <- .rm_int(model.matrix(~ noise, data = wine))
INT <- matrix(1, nrow = nrow(wine))

m(list(INT, X, Z))
loss <- k_ontram_loss(ncol(Y))
compile(m, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001))
cent <- metric_cqwk(ncol(Y))
compile(m, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 0.001),
metrics = c(cent))
fit(m, x = list(INT, X, Z), y = Y, batch_size = nrow(wine), epoch = 10,
    view_metrics = FALSE)

idx <- 8
loss(Y[idx, , drop = FALSE], m(list(INT[idx, , drop = FALSE],
     X[idx, , drop = FALSE], Z[idx, , drop = FALSE])))

tm <- Polr(rating ~ temp + contact + noise, data = wine)
logLik(tm, newdata = wine[idx,])

tmp <- get_weights(m)
tmp[[1]][] <- .to_gamma(coef(as.mlt(tm))[1:4])
tmp[[2]][] <- coef(tm)[1:2]
tmp[[3]][] <- coef(tm)[3]
set_weights(m, tmp)

loss(k_constant(Y), m(list(INT, X, Z)))
- logLik(tm)
- logLik(tm) / nrow(wine)

}
