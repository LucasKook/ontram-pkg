test_that("conversion from gamma to theta works", {
  expect_equal(.to_theta(c(-1, 0, 0)), c(-1, 0, 1))
})
