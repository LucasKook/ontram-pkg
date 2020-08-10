# Simple shift ordinal transformation neural network

set.seed(2410)

# Dependencies ------------------------------------------------------------

library(ontram)
library(tram)

# Data --------------------------------------------------------------------

data("wine", package = "ordinal")

# Preprocessing -----------------------------------------------------------

fml <- rating ~ temp + contact
x_train <- model.matrix(fml, data = wine)[, -1L]
y_train <- model.matrix(~ 0 + rating, data = wine)

# baseline model ----------------------------------------------------------

blm <- Polr(fml, data = wine)
preds <- predict(blm, type = "quantile", p = 0.5)
acc <- sum(wine$rating == preds)/nrow(wine) # 44.4 %
ce <- -as.numeric(logLik(blm))/nrow(wine) # 1.201

# non-parametric softmax --------------------------------------------------

model <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = ncol(x_train), activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = ncol(y_train), activation = "softmax")

model %>%
  compile(optimizer = "rmsprop", loss = "categorical_crossentropy",
          metrics = c("accuracy"))

n_epochs <- 1000

history <- model %>%
  fit(x_train, y_train, epochs = n_epochs, batch_size = 8, view_metrics = FALSE)

acc2 <- history$metrics$accuracy[n_epochs] # 44.4
ce2 <- history$metrics$loss[n_epochs] # 1.18

# ontram model ------------------------------------------------------------

mo <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
                  method = "logit", n_batches = 16, epochs = n_epochs)
fit_ontram(mo, x_train = x_train, y_train = y_train)

mo_pred <- predict(mo, x = x_train, y = y_train)

acc3 <- sum(mo_pred$response == wine$rating)/nrow(wine)
ce3 <- mo_pred$negLogLik

# Comparison --------------------------------------------------------------

comp <- rbind(c(acc, ce), c(acc2, ce2), c(acc3, ce3))
dimnames(comp) <- list(c("tram", "nn", "ontram"), c("accurcacy", "negLogLik"))
comp

comp2 <- rbind(coef(blm, with_baseline = TRUE), coef(mo, with_baseline = TRUE))
rownames(comp2) <- c("tram", "ontram")
comp2

# Include validation set --------------------------------------------------

idx <- sample(seq_len(nrow(wine)), 15)
x_valid <- x_train[idx,]
x_train2 <- x_train[-idx, ]
y_valid <- y_train[idx, ]
y_train2 <- y_train[-idx, ]

mo2 <- ontram_polr(x_dim = ncol(x_train), y_dim = ncol(y_train),
                  method = "logit", n_batches = 16, epochs = n_epochs/4)
mo2hist <- fit_ontram(mo2, x_train = x_train2, y_train = y_train2, history = TRUE,
                      x_test = x_valid, y_test = y_valid)
plot(mo2hist)


