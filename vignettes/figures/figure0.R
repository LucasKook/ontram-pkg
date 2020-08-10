library(ggplot2)
library(scales)
library(ggpubr)

data(wine, package = "ordinal")
dat <- data.frame(y = sort(unique(wine$rating)),
                  dens = c(table(wine$rating)/nrow(wine)),
                  distr = cumsum(c(table(wine$rating))/nrow(wine)))
p1 <- ggplot(dat) +
  geom_segment(aes(x = y, xend = y, y = 0, yend = dens)) +
  geom_point(aes(x = y, y = dens)) +
  labs(x = expression(italic(y[k])), y = expression(P(italic(Y==y[k])))) +
  theme_pubr()

p2 <- ggplot(dat) +
  geom_point(aes(x = y, y = distr)) +
  geom_step(aes(x = y, y = distr, group = NA)) +
  labs(x = expression(italic(y[k])), y = expression(P(italic(Y<=y[k])))) +
  ylim(0, 1) +
  theme_pubr()

(p3 <- ggarrange(p1, p2, labels = c("A", "B")))

ggsave("vignettes/figures/figure0.pdf", plot = p3, width = 10, height = 4)
ggsave("vignettes/figures/figure0.png", plot = p3, width = 10, height = 4)
