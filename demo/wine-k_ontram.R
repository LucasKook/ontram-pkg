# Wine demo k_ontram
# LK, Nov 2021

# Deps --------------------------------------------------------------------

library(tram)
library(ontram)

# Wine --------------------------------------------------------------------

data("wine", package = "ordinal")
wine$noise <- rnorm(nrow(wine))

mbl <- k_mod_baseline(5L, name = "baseline")
msh <- mod_shift(2L, name = "linear_shift")
mim <- mod_shift(1L, name = "complex_shift")
m <- k_ontram(mbl, list(msh, mim))

X <- .rm_int(model.matrix(~ temp + contact, data = wine))
Y <- model.matrix(~ 0 + rating, data = wine)
Z <- .rm_int(model.matrix(~ noise, data = wine))
INT <- matrix(1, nrow = nrow(wine))

loss <- k_ontram_loss(ncol(Y))
compile(m, loss = loss, optimizer = optimizer_adam(lr = 1e-2, decay = 0.001))
fit(m, x = list(INT, X, Z), y = Y, batch_size = nrow(wine), epoch = 10,
    view_metrics = FALSE)

tm <- Polr(rating ~ temp + contact + noise, data = wine)
logLik(tm, newdata = wine[idx,])

tmp <- get_weights(m)
tmp[[1]][] <- .to_gamma(coef(as.mlt(tm))[1:4])
tmp[[2]][] <- coef(tm)[1:2]
tmp[[3]][] <- coef(tm)[3]
set_weights(m, tmp)

loss(k_constant(Y), m(list(INT, X, Z)))
- logLik(tm)

debugonce(predict.k_ontram)
predict(m, list(INT, X, Z), type = "cumhaz")
