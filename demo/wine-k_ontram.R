# Wine demo k_ontram
# LK, Nov 2021

# Deps --------------------------------------------------------------------

library(tram)
library(ontram)

# Wine --------------------------------------------------------------------

data("wine", package = "ordinal")
wine$noise <- rnorm(nrow(wine))

X <- ontram:::.rm_int(model.matrix(~ temp + contact, data = wine))
Y <- model.matrix(~ 0 + rating, data = wine)
Z <- ontram:::.rm_int(model.matrix(~ noise, data = wine))
INT <- matrix(1, nrow = nrow(wine))

mbl <- k_mod_baseline(ncol(Y), name = "baseline")
msh <- mod_shift(ncol(X), name = "linear_shift")
mim <- mod_shift(ncol(Z), name = "complex_shift")
m <- k_ontram(mbl, list(msh, mim))

k_nll <- k_ontram_loss(ncol(Y))
loss <- k_ontram_rps(ncol(Y))
compile(m, loss = loss, optimizer = optimizer_adam(learning_rate = 1e-2, decay = 1e-4),
        metrics = c(
          metric_nll(ncol(Y)),
          metric_acc(ncol(Y)),
          metric_cqwk(ncol(Y)),
          metric_k_auc(ncol(Y)),
          metric_binll(ncol(Y))
        )
)
mh <- fit(m, x = list(INT, X, Z), y = Y, batch_size = ncol(Y), epoch = 3e2,
    view_metrics = FALSE, validation_split = 0.1)
plot(mh)

tm <- Polr(rating ~ temp + contact + noise, data = wine)

k_nll(k_constant(Y), m(list(INT, X, Z)))
- logLik(tm) / nrow(wine)

head(predict(m, list(INT, X, Z), type = "distribution"))

rps_polr <- function(y_true, y_pred) {
  K <- ncol(y_true)
  y_cum <- apply(y_true, 1, cumsum)
  briers <- (y_pred[, 1:(K - 1), drop = FALSE] -
               y_cum[, 1:(K - 1), drop = FALSE])^2
  mean(apply(briers, 2, mean))
}

tm_preds <- predict(tm, newdata = wine[, !colnames(wine) == "rating"],
                    type = "distribution")

loss(k_constant(Y), m(list(INT, X, Z)))
rps_polr(Y, tm_preds)

pROC::auc(wine$rating <= 3L, predict(tm, type = "distribution", newdata = wine[, !colnames(wine) == "rating"])[3,])
evaluate(m, list(INT, X, Z), Y, batch_size = ncol(Y))
