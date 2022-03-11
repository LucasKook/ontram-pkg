
dgp <- function(n = 100, ncl = 4) {
  data.frame(
    y = ordered(sample.int(ncl, n, replace = TRUE)),
    x = rnorm(n),
    z = rnorm(n)
  )
}

test_that("some ontram", {

  n <- 100
  ncl <- 4
  df <- dgp(n = n, ncl = ncl)

  mbl <- k_mod_baseline(ncl, name = "baseline")
  msh <- mod_shift(1L, name = "linear_shift")
  mcs <- keras_model_sequential() %>%
    layer_dense(input_shape = 1L, units = 4, activation = "relu") %>%
    layer_dense(units = 1L, use_bias = FALSE)

  m <- k_ontram(mod_baseline = mbl, list(msh, mcs))

  coef(m, which = "baseline")
  coef(m, which = "linear_shift")
  coef(m)

  loss <- k_ontram_loss(ncl)
  acc <- metric_acc(ncl)
  compile(m, optimizer = "adam", loss = loss, metrics = c(acc))

  y_train <- suppressWarnings(to_categorical(df$y)[, -1])
  INT <- matrix(1, nrow = nrow(y_train))
  x_train <- .rm_int(model.matrix(~ x, data = df))
  z_train <- .rm_int(model.matrix(~ z, data = df))
  INP <- list(INT, x_train, z_train)

  fit(m, x = INP, y = y_train, view_metrics = FALSE)

  evaluate(m, INP, y_train)

  lapply(c("distribution", "density", "trafo", "baseline_only", "hazard",
           "cumhazard", "survivor", "odds", "terms"), function(ttype) {
             preds <- predict(m, x = INP, type = ttype)
             expect_is(preds, "matrix")
             expect_false(any(is.nan(preds)))
             invisible(NULL)
           })
})
