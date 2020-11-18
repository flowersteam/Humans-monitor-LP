library(tidyverse)
library(gridExtra)
library(cowplot)

# Load data
df <- tbl_df(read.csv('data/coef_data.csv')) %>%
  dplyr::rename(wPC=pc_coef, wLP=rlp_coef) %>% 
  dplyr::mutate(A4minusA3 = -A3minusA4)

# Join datasets and optionally filter poorly fitted models
# df <- dplyr::filter(df, aic_diff >= 2)
df <- dplyr::inner_join(df, diffs, by='sid')

# Convert categorical variables to factors
df <- dplyr::mutate_at(df, vars(sid, grp, ntm), funs(factor))

# Fit linear models
lm_ <- lm(int_dif_A4A3 ~ grp*(wPC+wLP), data=df)
summary(lm_)
lm_ <- lm(int_dif_A3A1 ~ grp*(wPC+wLP), data=df)
summary(lm_)

# Set colors
df$grp <- recode(df$grp, '0' = 'IG', '1' = 'EG')
gcolors = c('#008fd5', '#fc4f30')

# Add empty axes with text only
gt0 <- ggplot() + geom_point() + theme_void() +  
  geom_text(aes(0, 0, label='Difficult vs. Easy'), size=5) + xlab(NULL) + 
  theme(
    panel.background = element_rect(fill = "white", color='white')
  )

gt1 <- ggplot() + geom_point() + theme_void() + 
  geom_text(aes(0, 0, label='Random vs. Difficult'), size=5) + xlab(NULL) + 
  theme(
    panel.background = element_rect(fill = "white", color='white')
  )


# (0, 0) wPC vs A3-A1
g00 <- ggplot(df, aes(x = wPC, y = A3minusA1, col = grp)) +
  geom_point(alpha=.4) +
  geom_smooth(method=lm, se=FALSE, fullrange = FALSE) +
  annotate('segment', x=-6, y=50, xend=-6, yend=240, arrow=arrow(length=unit(0.3, "cm"))) +
  annotate('text', x=-6.5, y=50, label='prefer HARDER', angle=90, size=2.5, hjust='left') + 
  annotate('segment', x=-6, y=-50, xend=-6, yend=-240, arrow=arrow(length=unit(0.3, "cm"))) +
  annotate('text', x=-6.5, y=-50, label='prefer EASIER', angle=90, size=2.5, hjust='right') +
  coord_cartesian(xlim=c(-4,4), ylim=c(-250,250), clip='off') +
  labs(
    title = bquote(italic(w[PC])),
    y = 'A3 - A1\n\n',
    col = 'Group') +
  scale_color_manual(values=gcolors) +
  theme_get() +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = 'gray', linetype = '11'),
    panel.grid.minor = element_line(color = NA),
    axis.line.x = element_line(color="black", size = .25),
    axis.line.y = element_line(color="black", size = .25),
    legend.position = 'none',
    axis.title.x = element_blank(),
    plot.title = element_text(size=20, face='bold.italic', hjust=.5)
  )

# (0, 1) wLP vs A3-A1
g01 <- ggplot(df, aes(x = wLP, y = A3minusA1, col = grp)) +
  geom_point(alpha=.4) +
  geom_smooth(method=lm, se=FALSE, fullrange = FALSE) +
  coord_cartesian(xlim=c(-4,5), ylim=c(-250,250)) +
  geom_smooth(method=lm, se=FALSE, fullrange=TRUE) +
  labs(
    title = bquote(italic(w[LP])),
    col = 'Group') +
  scale_color_manual(values=gcolors) + 
  theme_get() +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = 'gray', linetype = '11'),
    panel.grid.minor = element_line(color = NA),
    axis.line.x = element_line(color="black", size = .25),
    axis.line.y = element_line(color="black", size = .25),
    legend.position = "none",
    axis.title.y = element_blank(),
    axis.title.x = element_blank(),
    plot.title = element_text(size=20, face='bold.italic', hjust=.5),
    plot.margin=unit(c(0.2,.5,0,1),"cm")
  )

# (1, 0) wPC vs A4-A3
g10 <- ggplot(df, aes(x = wPC, y = A4minusA3, col = grp)) +
  geom_point(alpha=.4) +
  annotate('segment', x=-6, y=50, xend=-6, yend=240, arrow=arrow(length=unit(0.3, "cm"))) +
  annotate('text', x=-6.5, y=50, label='prefer HARDER', angle=90, size=2.5, hjust='left') + 
  annotate('segment', x=-6, y=-50, xend=-6, yend=-240, arrow=arrow(length=unit(0.3, "cm"))) +
  annotate('text', x=-6.5, y=-50, label='prefer EASIER', angle=90, size=2.5, hjust='right') + 
  geom_smooth(method=lm, se=FALSE, fullrange = FALSE) +
  coord_cartesian(xlim=c(-4,4), ylim=c(-250,250), clip='off') +
  labs(
    y = 'A4 - A3\n\n',
    x = bquote(italic(w[PC])~(italic(z)-score)),
    col = 'Group') +
  scale_color_manual(values=gcolors) +
  theme_get() +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = 'gray', linetype = '11'),
    panel.grid.minor = element_line(color = NA),
    axis.line.x = element_line(color="black", size = .25),
    axis.line.y = element_line(color="black", size = .25),
    legend.position = "none"
  )

# (1, 1) wLP vs A4-A3
g11 <- ggplot(df, aes(x = wLP, y = A4minusA3, col = grp)) +
  geom_point(alpha=.4) +
  geom_smooth(method=lm, se=FALSE, fullrange = FALSE) +
  coord_cartesian(xlim=c(-4,5), ylim=c(-250,250)) +
  geom_smooth(method=lm, se=FALSE, fullrange=TRUE) +
  labs(
    x = bquote(italic(w[LP])~(italic(z)-score)),
    col = 'Group') +
  scale_color_manual(values=gcolors) + theme_get() +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = 'gray', linetype = '11'),
    panel.grid.minor = element_line(color = NA),
    axis.line.x = element_line(color="black", size = .25),
    axis.line.y = element_line(color="black", size = .25),
    legend.position = c(-1.8, 1),
    legend.key=element_blank(),
    axis.title.y = element_blank(),
    plot.margin=unit(c(.2,.5,.2,1),"cm")
  )


pg <- arrangeGrob(gt0, g00, g01, gt1, g10, g11, ncol=3, nrow=2)
plot(pg)

# ggsave(
#   'figure4b.svg',
#   plot = pg,
#   path = '../figures/',
#   width = 10,
#   height = 5,
#   units = 'in',
#   dpi = 300
# )