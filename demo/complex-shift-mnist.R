# Complex image shift effect
# F_Y(y|img) = F_Z(y^T theta - beta(img))

set.seed(2410)

# Dependencies ------------------------------------------------------------

library(ontram)
library(tram)
library(ggplot2)

# Data mnist --------------------------------------------------------------

mnist <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
y_train <- to_categorical(y_train)
y_test <- to_categorical(y_test)
x_train <- array_reshape(x_train, c(60000, 28, 28, 1))
x_test <- array_reshape(x_test, c(10000, 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255
nim <- 30000
x_train <- x_train[1:nim, , , , drop = FALSE]
y_train <- y_train[1:nim, , drop = FALSE]

# ontram model ------------------------------------------------------------

mbl <- mod_baseline(ncol(y_train))
mim <-  keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

mo <- ontram(mod_bl = mbl, mod_sh = NULL, mod_im = mim, method = "logit",
             n_batches = 300, epochs = 10, x_dim = 1, img_dim = c(28, 28, 1),
             y_dim = ncol(y_train))
fit_ontram(mo, x_train = NULL, y_train = y_train, img_train = x_train)

# Evaluation --------------------------------------------------------------

# Train
preds <- predict(mo, NULL, y_train, x_train)
(confmat <- table(preds$response - 1, mnist$train$y[1:nim]))
(acc <- sum(preds$response - 1 == mnist$train$y[1:nim])/nim)

# Test
preds_test <- predict(mo, NULL, y_test, x_test)
(confmat_test <- table(preds_test$response - 1, mnist$test$y))
(acc_test <- sum(preds_test$response - 1 == mnist$test$y)/nrow(y_test))

# Interpretability --------------------------------------------------------

# Baseline trafo coef
cfb <- get_weights(mo$mod_baseline)[[1]] %>% diag %>% ontram:::.to_theta()

# Train
dat_train <- data.frame(class = ordered(mnist$train$y[1:nim]),
                        logors = mo$mod_image(x_train)$numpy())
ggplot(dat_train) +
  geom_histogram(aes(x = logors, fill = class), bins = 200) +
  labs(x = "estimated log odds ratios",
       subtitle = paste0("Training data accuracy ", round(acc, 3))) +
  theme_bw()

ggplot(dat_train) +
  geom_boxplot(aes(y = logors, x = class, fill = class, color = class)) +
  labs(y = "estimated log odds ratios", x = "class",
       subtitle = paste0("Training data accuracy ", round(acc, 3))) +
  theme_bw()

# Test
dat_test <- data.frame(class = ordered(mnist$test$y),
                       logors = mo$mod_image(x_test)$numpy())
ggplot(dat_test) +
  geom_histogram(aes(x = logors, fill = class), bins = 200) +
  labs(x = "estimated log odds ratios",
       subtitle = paste0("Test data accuracy ", round(acc_test, 3))) +
  theme_bw()

ggplot(dat_test) +
  geom_boxplot(aes(y = logors, x = class, fill = class, color = class)) +
  labs(y = "estimated log odds ratios", x = "class",
       subtitle = paste0("Test data accuracy ", round(acc_test, 3))) +
  theme_bw()

# Include validation set --------------------------------------------------

idx <- sample(seq_len(nrow(y_train)), floor(nrow(y_train) * 0.2))
x_valid <- x_train[idx, , , , drop = FALSE]
x_train2 <- x_train[-idx, , , , drop = FALSE]
y_valid <- y_train[idx, , drop = FALSE]
y_train2 <- y_train[-idx, , drop = FALSE]

mo2 <- ontram(mod_bl = mbl, mod_sh = NULL, mod_im = mim, method = "logit",
             n_batches = 300, epochs = 10, x_dim = 1, img_dim = c(28, 28, 1),
             y_dim = ncol(y_train))
mo2hist <- fit_ontram(mo2, x_train = NULL, y_train = y_train2, img_train = x_train2,
                      history = TRUE, x_test = NULL, y_test = y_valid, img_test = x_valid)
plot(mo2hist)
