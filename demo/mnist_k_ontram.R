# Complex image shift effect
# F_Y(y|img) = F_Z(y^T theta - beta(img))
# k_ontram

set.seed(2410)

# Dependencies ------------------------------------------------------------

library(ontram)

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

mbl <- k_mod_baseline(ncol(y_train))
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
m <- k_ontram(mbl, mim)

loss <- k_ontram_loss(ncol(y_train))
compile(m, loss = loss, optimizer = optimizer_adam(lr = 1e-4))
fit(m, x = list(matrix(1, nrow(y_train)), x_train),
    y = y_train, validation_split = 0.3, epoch = 5, batch_size = 32)

preds <- predict(m, x = list(matrix(1, nrow(y_test)), x_test))
trafo <- preds[,1:9] - do.call("cbind", lapply(1:9, \(x) preds[,10]))
pdf <- apply(cbind(plogis(trafo), 1), 1, diff)

matplot(pdf, type = "l")
