# First thoughts on visualizing the transformation function
# For unconditional transformation models


# Dependencies ------------------------------------------------------------

library(tram)
library(tidyverse)
library(ggplot2)
library(ggpubr)
theme_set(theme_pubr())

# Data --------------------------------------------------------------------

data("faithful", package = "datasets")
data("wine", package = "ordinal")

# The model ---------------------------------------------------------------

m <- Colr(waiting ~ 1, data = faithful, order = 10, add = 2)

# Visualization -----------------------------------------------------------

setup <- function(object, res = 1e3, continuous = TRUE, nbin = 10, add = 1) {
  y <- variable.names(object, "response")
  qs <- mkgrid(object, n = res)[[y]]
  cdf <- predict(object, type = "distribution", q = qs)
  pdf <- predict(object, type = "density", q = qs)
  trafo <- predict(object, type = "trafo", q = qs)
  if (continuous) {
    pdfz <- object$model$todistr$d(trafo)
  } else {
    pdfz <- diff(c(0, object$model$todistr$p(trafo)))
  }
  dat <- data.frame(y = qs, cdf = cdf, pdf = pdf, trafo = trafo, pdfz = pdfz)
  if (is.numeric(qs))
    dat$cols <- cut(qs, quantile(qs, probs = seq(0, 1, length.out = nbin)))
  if (is.factor(qs)) {
    trafo2 <- seq(min(trafo) - add, max(trafo[is.finite(trafo)]) + add,
               length.out = res)
    pdfz2 <- object$model$todistr$d(trafo2)
    cols2 <- cut(trafo2, breaks = c(-Inf, trafo))
    aux <- data.frame(trafo2 = trafo2, pdfz2 = pdfz2, cols2 = cols2)
    ret <- list(dat, aux)
    return(ret)
  }
  return(dat)
}

trafo_plot <- function(setup, continuous = TRUE, fill = FALSE, zlim = c(-4, 4),
                       tag = NULL) {

  p0 <- ggplot() + theme_void()

  if (!is.null(tag))
    p0 <- p0 + geom_text(aes(x = 0, y = 0, label = tag))

  if (continuous) {
    p1 <- ggplot(setup) +
      geom_line(aes(x = y, y = trafo, color = y)) +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.title.y = element_blank(), axis.ticks = element_blank(),
            axis.text = element_blank()) +
      ylim(zlim)

    p2 <- ggplot(setup) +
      geom_line(aes(x = trafo, y = pdfz, color = y)) +
      coord_flip() +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.x = element_blank()) +
      xlim(zlim)

    p3 <- ggplot(setup) +
      geom_line(aes(x = y, y = pdf, color = y)) +
      scale_y_reverse() +
      theme(legend.position = "none", axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.y = element_blank())
    p1 <- p1 + scale_color_viridis_c()
    p2 <- p2 + scale_color_viridis_c()
    p3 <- p3 + scale_color_viridis_c()
    if (fill) {
      p2 <- p2 + geom_area(aes(x = trafo, y = pdfz, fill = cols)) + scale_fill_viridis_d()
      p3 <- p3 + geom_area(aes(x = y, y = pdf, fill = cols)) + scale_fill_viridis_d()
    }
  } else {
    sup2 <- setup[[1]]
    sup2$trafo[!is.finite(sup2$trafo)] <- NA
    p1 <- ggplot(sup2) +
      # geom_segment(aes(x = y, y = trafo, xend = y, yend = -6.595), color = "gray", lty = 2) +
      # geom_segment(aes(x = y, y = trafo, xend = 1, yend = trafo), color = "gray", lty = 2) +
      geom_point(aes(x = y, y = trafo, color = y)) +
      geom_text(aes(x = y, y = trafo, label = paste0("italic(h(y[", 1:5, "]~'|'~x))")), parse = TRUE, nudge_x = 0.5, nudge_y = -0.5) +
      theme(legend.position = "none", axis.title.x = element_blank(),
            axis.ticks = element_blank(), axis.title.y = element_blank(),
            axis.text = element_blank()) +
      ylim(range(setup[[2]]$trafo2))

    p2 <- ggplot(setup[[2]]) +
      geom_line(aes(x = trafo2, y = pdfz2)) +
      geom_area(aes(x = trafo2, y = pdfz2, fill = cols2), show.legend = FALSE) +
      coord_flip() +
      scale_y_reverse() +
      theme(legend.position = "none", #axis.title.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.text.x = element_blank()) +
      xlim(range(setup[[2]]$trafo2)) +
      xlab(expression(italic(z))) +
      ylab(expression(italic(f[Z](z))))

    p3 <- ggplot(setup[[1]]) +
      geom_segment(aes(x = y, xend = y, y = 0, yend = pdf, color = y)) +
      geom_point(aes(x = y, y = pdf, color = y)) +
      scale_y_reverse() +
      theme(legend.position = "none", #axis.title.y = element_blank(),
            axis.ticks.y = element_blank(), axis.text.y = element_blank()) +
      labs(x = expression(italic(y[k])),
           y = expression(italic(f[Y](y[k]~"|"~x))))
    p1 <- p1 + scale_fill_viridis_d()
    p2 <- p2 + scale_fill_viridis_d()
    p3 <- p3 + scale_color_viridis_d()
  }
  ggarrange(p2, p1, p0, p3, ncol = 2, nrow = 2, align = "hv")
}

# Discrete Model ----------------------------------------------------------

data("wine", package = "ordinal")
m1 <- Polr(rating ~ 1, data = wine, method = "logistic")
dat1 <- setup(m1, continuous = FALSE, add = 3)
(out1 <- trafo_plot(setup = dat1, continuous = FALSE))

# Discrete Model ----------------------------------------------------------

data("wine", package = "ordinal")
m2 <- Polr(rating ~ 1, data = wine, method = "cloglog")
dat2 <- setup(m2, continuous = FALSE, add = 3)
(out2 <- trafo_plot(setup = dat2, continuous = FALSE))

# Save plots --------------------------------------------------------------

# ggsave(out, filename = "continuous.svg")
# ggsave(outf, filename = "continuous_filled.svg")
ggsave(out1, filename = "vignettes/figures/discrete.pdf")
ggsave(out2, filename = "vignettes/figures/discrete-cloglog.pdf")
ggsave(out1, filename = "vignettes/figures/discrete.png")
ggsave(out2, filename = "vignettes/figures/discrete-cloglog.png")












