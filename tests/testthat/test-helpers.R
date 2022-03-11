
test_that("conversion from gamma to theta works", {
  expect_warning(.to_gamma(c(1, -1, -2))) # because input has to be increasing
  expect_equal(.to_gamma(.to_theta(c(-1, 0, 0))), c(-1, 0, 0))
})

