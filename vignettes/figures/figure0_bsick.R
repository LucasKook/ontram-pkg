# First thoughts on visualizing the transformation function
# For unconditional transformation models


# Dependencies ------------------------------------------------------------

library(tram)
library(tidyverse)
library(ggplot2)
library(ggpubr)
theme_set(theme_pubr())

# # Data --------------------------------------------------------------------

data("wine", package = "ordinal")
table(wine$rating)/nrow(wine)

# Discrete Model ----------------------------------------------------------

m1 <- Polr(rating ~ 1, data = wine, method = "logistic")

# Visualization -----------------------------------------------------------
res <- 1e3
add <- 3
object <- m1
y <- variable.names(object, "response")
(qs <- mkgrid(object, n = res)[[y]])  # y-values
(cdf <- predict(object, type = "distribution", q = qs))
(pdf <- predict(object, type = "density", q = qs))  # lik-contributions
(trafo <- predict(object, type = "trafo", q = qs))  # z-values
(pdfz <- diff(c(0, object$model$todistr$p(trafo))))  # lik-contributions

dat <- data.frame(y = qs, cdf = cdf, pdf = pdf, trafo = trafo, pdfz = pdfz)
dat$xaxis <- sprintf("y[%i]", dat$y)
dat$xaxis <- paste0(dat$y, "\n", dat$xaxis)
dat$y <- as.numeric(qs)
dat$ystart <- c(0, dat$cdf[1:4])
dat$xend <- dat$y +1
dat$col <- factor(dat$y)

# make grid on z-skale with res=1000 points in range of z-values plus add-margin
trafo2 <- seq(min(trafo) - add, max(trafo[is.finite(trafo)]) + add, length.out = res)
# determine densitity on the z-grid
pdfz2 <- object$model$todistr$d(trafo2)  # same as dlogis(trafo2)
# produce z-intervals corresponding to the taken z-values and -Inf, +Inf at border
cols2 <- cut(trafo2, breaks = c(-Inf, trafo))
# determine CDF on z-grid
cdfz2 <- plogis(trafo2)

aux <- data.frame(trafo2 = trafo2, pdfz2 = pdfz2, cols2 = cols2, cdfz2 = cdfz2)

p0 <- ggplot() + theme_void()
p0 <- p0 + geom_text(aes(x = 0, y = 0, label = "wine"))

p1 <- ggplot(dat) +
geom_point(aes(x = y, y = trafo, color = y)) +
theme(legend.position = "none", axis.title.x = element_blank(),
  axis.title.y = element_blank(), axis.ticks = element_blank(),
  axis.text = element_blank()) +
ylim(range(aux$trafo2))

p2 <- ggplot(aux) +
geom_line(aes(x = trafo2, y = pdfz2)) +
geom_area(aes(x = trafo2, y = pdfz2, fill = cols2), show.legend = FALSE) +
coord_flip() +
scale_y_reverse() +
theme(legend.position = "none", axis.title.x = element_blank(),
  axis.ticks.x = element_blank(),
  axis.text.x = element_blank()) +
xlim(range(aux$trafo2)) +
xlab("z = h(y)")


p3 <- ggplot(dat) +
geom_segment(aes(x = y, xend = y, y = 0, yend = pdf, color = col)) +
geom_point(aes(x = y, y = pdf, color = col)) +
scale_y_reverse() +
theme(legend.position = "none", axis.title.y = element_blank(),
  axis.ticks.y = element_blank(), axis.text.y = element_blank())

p1 <- p1 + scale_fill_viridis_d()
p2 <- p2 + scale_fill_viridis_d()
p3 <- p3 + scale_color_viridis_d()
ggarrange(p2, p1, p0, p3, ncol = 2, nrow = 2)


#--- f_y(y) und f_z(z) ---------

pfy <- ggplot(dat) +
  geom_segment(aes(x = y, xend = y, y = 0, yend = pdf, color = col)) +
  geom_point(aes(x = y, y = pdf, color = col)) +
  xlab(expression(italic(y[k]))) + ylab(expression(italic(f[Y](y[k]~"|"~x)))) +
  theme(legend.position = "none")+
  scale_y_continuous(breaks = round(c(0, dat$pdf), 2), limits = c(0, 0.4))

pfz <- ggplot(aux) +
  geom_line(aes(x = trafo2, y = pdfz2)) +
  geom_area(aes(x = trafo2, y = pdfz2, fill = cols2), show.legend = FALSE) +
  xlab(expression(italic(z))) + ylab(expression(italic(f[Z](z)))) +
  scale_x_continuous(breaks = round(dat$trafo[1:4], 1),
                     labels = parse(text = paste0("italic(h[", 1:4, "])")))


#--- F_y(y) und F_z(z) ---------

# F_y(y)

pFy <- ggplot(dat) +
  geom_hline(yintercept = cdf, linetype = "dashed", color = "darkgrey") +
  geom_segment(aes(x = y, y = cdf, xend = xend, yend = cdf), size = 1.2, show.legend = FALSE, color = "gray") +
  geom_segment(aes(x = y, y = ystart, xend = y, yend = cdf, color = col), size = 1.2, show.legend = FALSE) +
  geom_point(aes(x = y, y = cdf, color = col), show.legend = FALSE) +
  xlab(expression(italic(y[k]))) + ylab(expression(italic(F[Y](y[k]~"|"~x)))) +
  coord_cartesian(xlim=c(1,5)) +
  scale_y_continuous(breaks=round(c(0, dat$cdf),2), limits=c(0,1))

#--- F_z(z) ---
dat4 <- dat[1:4,]
dat$tstart <- c(min(aux$trafo2), dat$trafo[1:4])
dat$trafo[5] <- max(aux$trafo2)

pFz <- ggplot(aux) +
  geom_hline(yintercept=dat$cdf, linetype="dashed", color = "darkgrey") +
  geom_step(data = aux, mapping = aes(x = trafo2, y = cdfz2)) +
  geom_segment(data = dat, aes(x = tstart, y = ystart, xend = trafo, yend = ystart, color = col),
               size = 1.2) +
  geom_segment(data = dat[-5,], aes(x = trafo, y = ystart, xend = trafo, yend = cdf, color = col),
               size = 1.2, show.legend = FALSE) +
  geom_point(data = dat4, mapping = aes(x = trafo, y = cdf, color = col), show.legend = FALSE) +
  xlab(expression(italic(z))) + ylab(expression(F[z](z))) +
  scale_y_continuous(breaks = round(c(0, dat$cdf), 2), limits = c(0, 1))+
  scale_x_continuous(breaks = round(dat$trafo[1:4], 1),
                     labels = parse(text = paste0("italic(h)[", 1:4, "]")))

pfy <- pfy + scale_color_viridis_d()
pfz <- pfz + scale_fill_viridis_d()
pFy <- pFy + scale_color_viridis_d(guide = "none")
pFz <- pFz + scale_color_viridis_d(guide = "none")

fig0 <- ggarrange(pfy, pfz, pFy, pFz, ncol = 2, nrow = 2, labels = c("A", "C", "B", "D"))
fig0


# Save plots --------------------------------------------------------------

ggsave(fig0, filename = "ordinal.png", width = 8, height = 6)
ggsave(fig0, filename = "ordinal.pdf", width = 8, height = 6)












