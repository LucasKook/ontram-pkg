library(ggplot2)
library(scales)
library(ggpubr)
library(tram)
library(patchwork)

theme_set(ggpubr::theme_pubr())

data("wine", package = "ordinal")
m1 <- Polr(rating ~ temp, data = wine, method = "logistic")

nd <- data.frame(temp = factor(c("cold", "warm")))

trafo <- predict(m1, newdata = nd, type = "trafo")
dens <- predict(m1, newdata = nd, type = "density")
distr <- predict(m1, newdata = nd, type = "distribution")

dat <- data.frame(
  y = ordered(rep(1:5, 2)),
  x = factor(rep(0:1, each = 5)),
  trafo = c(trafo), dens = c(dens), distr = c(distr),
  shift = c(coef(m1))
)

dat[!is.finite(dat$trafo),"trafo"] <- NA
add <- 3; res <- 1e3
trafo2 <- seq(min(trafo) - add, max(trafo[is.finite(trafo)]) + add,
              length.out = res)
pdfz2 <- m1$model$todistr$d(trafo2)
cols2 <- cut(trafo2, breaks = c(-Inf, trafo[,1]))
cols3 <- cut(trafo2, breaks = c(-Inf, trafo[,2]))
aux <- data.frame(trafo2 = trafo2, pdfz2 = pdfz2, cols2 = cols2, cols3 = cols3)

p1 <- ggplot(dat) +
  geom_path(aes(x = y, y = trafo, group = y), arrow = arrow(type = "closed",
                                                            angle = 20,
                                                            length = ggplot2::unit(0.3, "cm"))) +
  geom_point(aes(x = y, y = trafo, color = x)) +
  geom_text(aes(x = 1.15, y = (trafo[1] + trafo[6]) / 2, label = "beta"), parse = TRUE) +
  geom_text(aes(x = 2.15, y = (trafo[2] + trafo[7]) / 2, label = "beta"), parse = TRUE) +
  geom_text(aes(x = 3.15, y = (trafo[3] + trafo[8]) / 2, label = "beta"), parse = TRUE) +
  geom_text(aes(x = 4.15, y = (trafo[4] + trafo[9]) / 2, label = "beta"), parse = TRUE) +
  labs(x = expression(italic(y[k])), y = expression(italic(h(y[k]~"|"~x))))


p2 <- ggplot(dat) +
  # geom_rug(aes(x = y, color = y), cex = 4) +
  geom_col(aes(x = y, y = dens, fill = x), position = position_dodge(width = 0.15),
           width = 0.1, show.legend = FALSE) +
  labs(y = expression(italic(f[Y](y[k]~"|"~x))), x = expression(italic(y[k])))

p3 <- ggplot(aux) +
  geom_line(aes(x = trafo2, y = pdfz2)) +
  geom_area(aes(x = trafo2, y = pdfz2, fill = cols2), show.legend = TRUE) +
  scale_fill_viridis_d(name = parse(text = "italic(y[k])"), labels = 1:5) +
  labs(x = expression(italic(z)), y = expression(italic(f[Z](z)))) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  geom_text(aes(x = -4, y = 0.2, label = "italic(x==0)"), parse = TRUE)

p4 <- ggplot(aux) +
  geom_line(aes(x = trafo2, y = pdfz2)) +
  geom_area(aes(x = trafo2, y = pdfz2, fill = cols3), show.legend = FALSE) +
  scale_fill_viridis_d() +
  labs(x = expression(italic(z)), y = expression(italic(f[Z](z)))) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  geom_text(aes(x = -4, y = 0.2, label = "italic(x==1)"), parse = TRUE)

(pp <- p2 | p1 | (p3 / p4) )

ggsave("vignettes/figures/conditional-polr.pdf", plot = pp, width = 10, height = 4)
ggsave("vignettes/figures/conditional-polr.png", plot = pp, width = 10, height = 4)
