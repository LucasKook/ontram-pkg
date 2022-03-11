
test_that("accuracy works", {
  acc <- k_ontram_acc(3L)
  obs <- matrix(c(1, 0, 0,
                  0, 1, 0),
                nrow = 2, byrow = TRUE)
  preds <- matrix(c(2, 3, 0,
                    2, 3, 0),
                  nrow = 2, byrow = TRUE)
  expect_equal(acc(k_constant(obs), k_constant(preds))$numpy(), 0.5)
})
