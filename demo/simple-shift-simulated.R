# Simulation study on simple ordinal transformation neural networks

set.seed(2410)

# Dependencies ------------------------------------------------------------

library(ontram)
library(tram)

# Simulate data -----------------------------------------------------------

n <- 1000
p <- 5
K <- 5
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
beta <- runif(p, min = -1, max = 1)
thetas <- sort(runif(K - 1, min = -4, max = 4))
linpred <- X %*% beta
mat_theta <- matrix(rep(thetas, n), nrow = n, ncol = K - 1, byrow = TRUE)
mat_linpred <- matrix(rep(linpred, K - 1), nrow = n, ncol = K -1)
cdfs <- cbind(0, plogis(mat_theta - mat_linpred), 1)
dens <- t(apply(cdfs, 1, diff))
Y <- t(apply(dens, 1, function(ps) rmultinom(1, 1, ps)))

# Baseline model and sanity checks ----------------------------------------

dat <- data.frame(y = ordered(apply(Y, 1, which.max)), x = X)
m <- Polr(y ~ ., data = dat, method = "logistic")
mce <- - c(logLik(m))/nrow(dat)
mcoef <- coef(m, with_baseline = TRUE)

# ontram model ------------------------------------------------------------

nn <- ontram_polr(x_dim = p, y_dim = K, n_batches = 200, epochs = 100, method = "logit")
fit_ontram(nn, x_train = X, y_train = Y)
nnce <- predict(nn, X, Y)$negLogLik
nncoef <- coef(nn, with_baseline = TRUE)

# Comparison --------------------------------------------------------------

# logLik
mce; nnce

# coefs
rbind(mcoef, nncoef)

