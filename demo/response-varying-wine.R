# Response-varying effects and shift effects in tabular data
# F_Y(y|img) = F_Z(y^T theta(img) - x^T beta)

set.seed(2410)

# Dependencies ------------------------------------------------------------

library(ontram)
library(tram)

# Data --------------------------------------------------------------------

data("wine", package = "ordinal")
wine$noise <- rnorm(nrow(wine), sd = 0.3) + as.numeric(wine$rating)

# Preprocessing -----------------------------------------------------------

fml <- rating ~ contact + temp
x_train <- model.matrix(fml, data = wine)[, -1L, drop = FALSE]
im_train <- model.matrix(rating ~ noise, data = wine)[, -1L, drop = FALSE]
y_train <- model.matrix(~ 0 + rating, data = wine)

# baseline model ----------------------------------------------------------

blm <- Polr(rating | noise ~ contact + temp, data = wine)
preds <- predict(blm, type = "quantile", p = 0.5)
acc <- sum(wine$rating == preds)/nrow(wine)
ce <- -as.numeric(logLik(blm))/nrow(wine)

# ontram model ------------------------------------------------------------

n_epochs <- 1000
mbl <- keras_model_sequential() %>%
  layer_dense(units = 4, input_shape = 1L, use_bias = TRUE, activation = "tanh") %>%
  layer_dense(units = 4, use_bias = TRUE)

msh <- mod_shift(ncol(x_train))

mo <- ontram(mod_bl = mbl, mod_sh = msh, method = "logit", n_batches = 10,
             epochs = n_epochs, x_dim = 1L, y_dim = ncol(y_train),
             response_varying = TRUE)
hist(mo$mod_baseline(im_train)$numpy(), breaks = nrow(wine))
fit_ontram(mo, x_train = x_train, y_train = y_train, img_train = im_train)
hist(mo$mod_baseline(im_train)$numpy(), breaks = nrow(wine))

# Evaluation --------------------------------------------------------------

preds1 <- predict(mo, x = x_train, y = y_train, im = im_train)
acc1 <- sum(preds1$response == wine$rating)/nrow(wine)
ce1 <- preds1$negLogLik

coef(blm, with_baseline = TRUE)
coef(mo)

matplot(t(preds1$pdf), type = "l")

# Include validation set --------------------------------------------------

idx <- sample(seq_len(nrow(wine)), 15)
x_valid <- x_train[idx, , drop = FALSE]
x_train2 <- x_train[-idx, , drop = FALSE]
y_valid <- y_train[idx, , drop = FALSE]
y_train2 <- y_train[-idx, , drop = FALSE]
im_valid <- im_train[idx, , drop = FALSE]
im_train2 <- im_train[-idx, , drop = FALSE]

mo2 <- ontram(mod_bl = mbl, mod_sh = msh, method = "logit", n_batches = 10,
             epochs = n_epochs/4, x_dim = 1L, y_dim = ncol(y_train),
             response_varying = TRUE)
mo2hist <- fit_ontram(mo, x_train = x_train2, y_train = y_train2, img_train = im_train2,
                      history = TRUE, x_test = x_valid, y_test = y_valid, img_test = im_valid)
plot(mo2hist)

