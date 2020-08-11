# Response-varying effects of image and simple clinical shifts
# F_Y(y|img) = F_Z(y^T theta(img) - x^T beta)

set.seed(2410)

# Dependencies ------------------------------------------------------------

library(ontram)
library(tram)

# Data mnist --------------------------------------------------------------

mnist <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
y_train <- to_categorical(y_train)
x_train <- array_reshape(x_train, c(60000, 28, 28, 1))
x_train <- x_train / 255
nim <- 30000
x_train <- x_train[1:nim, , , , drop = FALSE]
y_train <- y_train[1:nim, , drop = FALSE]

# x_tab <- matrix(rnorm(nrow(y_train) * 2), ncol = 2)

# ontram model ------------------------------------------------------------

mbl <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1),
                use_bias = TRUE) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 9)
# msh <- mod_shift(ncol(x_tab))

mo <- ontram(mod_bl = mbl, mod_sh = NULL, method = "logit", n_batches = 300,
             epochs = 5, x_dim = c(28, 28), y_dim = ncol(y_train),
             response_varying = TRUE)
hist(mbl(x_train)$numpy(), breaks = 1e3)
fit_ontram(mo, x_train = NULL, y_train = y_train, img_train = x_train)
hist(mbl(x_train)$numpy(), breaks = 1e3)

# Evaluation --------------------------------------------------------------

nim <- 1000
x_eval <- x_train[1:nim, , , , drop = FALSE]
y_eval <- y_train[1:nim, , drop = FALSE]
preds <- predict(mo, NULL, y_eval, im = x_eval)
acc <- sum(preds$response - 1 == mnist$train$y[1:nim])/nim
confmat <- table(preds$response - 1, mnist$train$y[1:nim])
matplot(t(preds$pdf), type = "l")

# Include validation set --------------------------------------------------

idx <- sample(seq_len(nrow(y_train)), floor(nrow(y_train) * 0.2))
x_valid <- x_train[idx, , , , drop = FALSE]
x_train2 <- x_train[-idx, , , , drop = FALSE]
y_valid <- y_train[idx, , drop = FALSE]
y_train2 <- y_train[-idx, , drop = FALSE]

mo2 <- ontram(mod_bl = mbl, mod_sh = NULL, method = "logit", n_batches = 300,
              epochs = 5, x_dim = c(28, 28), y_dim = ncol(y_train),
              response_varying = TRUE)
mo2hist <- fit_ontram2(mo2, x_train = NULL, y_train = y_train2, img_train = x_train2,
                       history = TRUE, x_test = NULL, y_test = y_valid, img_test = x_valid)
plot(mo2hist)
