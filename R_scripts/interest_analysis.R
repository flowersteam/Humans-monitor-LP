library(tidyverse)
library(gridExtra)

# Get interest difference
dft <- tbl_df(read.csv('data/post_play_long.csv')) %>% 
  dplyr::select(sid,grp,tid,int)
A1 <- dplyr::filter(dft, tid == 1)
A3 <- dplyr::filter(dft, tid == 3)
A4 <- dplyr::filter(dft, tid == 4)
rm(dft)
A3['int_dif_A4A3'] = A4$int - A3$int
A3['int_dif_A3A1'] = A3$int - A1$int
diffs <- dplyr::select(A3, sid, int_dif_A4A3, int_dif_A3A1)
rm(A3, A4)

# Get coefficients data
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

# Define looks
df$grp <- recode(df$grp, '0' = 'IG', '1' = 'EG')
gcolors = c('#008fd5', '#fc4f30')

# A1. Plot wPC
g_int_wPC_1 <- ggplot(df, aes(x = wPC, y = int_dif_A4A3, col = grp)) +  
  geom_point(alpha=.7) + scale_color_manual(values=gcolors) + 
  geom_smooth(method=lm, se=FALSE, fullrange = TRUE) +
  scale_x_continuous(expand=c(0,0), limits=c(-10,10)) +
  scale_y_continuous(expand=c(0,0), limits=c(-10,10)) +
  coord_cartesian(xlim=c(-3,3.5), ylim=c(-10,10)) +
  labs(
    y = 'Interest rating (A4 vs. A3)',
    x = bquote(italic(w[PC])~(italic(z)-score)),
    col = 'Group') +
  theme_set(
    theme(
      panel.background = element_blank(),
      panel.grid.major = element_line(color = 'gray', linetype = '11'),
      panel.grid.minor = element_line(color = NA),
      axis.line.x = element_line(color="black", size = .25),
      axis.line.y = element_line(color="black", size = .25),
      legend.key = element_rect(color=NA, fill=alpha('white', .5)),
      legend.position = c(0.01, 0.01), 
      legend.justification = c(0, 0),
      legend.background = element_rect(fill=alpha('white', 0.4), size=0)
    )
  )
# A2. Plot wLP
g_int_wLP_1 <- ggplot(df, aes(x = wLP, y = int_dif_A4A3, col = grp)) +  
  geom_point(alpha=.7) + scale_color_manual(values=gcolors) +
  geom_smooth(method=lm, se=FALSE, fullrange=TRUE) +
  scale_x_continuous(expand=c(0,0), limits=c(-10,10)) +
  scale_y_continuous(expand=c(0,0), limits=c(-10,10)) +
  coord_cartesian(xlim=c(-4,5), ylim=c(-10,10)) +
  labs(
    x = bquote(italic(w[LP])~(italic(z)-score)),
    col = 'Group') +
  theme_get() + theme(
    legend.position = "none",
    axis.title.y = element_blank()
  )

plotA <- arrangeGrob(g_int_wPC_1, g_int_wLP_1, ncol=2)

ggsave(
  'intpref_A4A3.png',
  plot = plotA,
  path = '/Users/alexten/Desktop',
  width = 7,
  height = 3,
  units = 'in',
  dpi = 100
)


# A1. Plot wPC
g_int_wPC_2 <- ggplot(df, aes(x = wPC, y = int_dif_A3A1, col = grp)) +  
  geom_point(alpha=.7) + scale_color_manual(values=gcolors) + 
  geom_smooth(method=lm, se=FALSE, fullrange = TRUE) +
  # scale_x_continuous(expand=c(0,0), limits=c(-10,10)) +
  # scale_y_continuous(expand=c(0,0), limits=c(-10,10)) +
  # coord_cartesian(xlim=c(-3,3.5), ylim=c(-10,10)) +
  labs(
    y = 'Interest rating (A3 - A1)',
    x = bquote(italic(w[PC])~(italic(z)-score)),
    col = 'Group') +
  theme_set(
    theme(
      panel.background = element_blank(),
      panel.grid.major = element_line(color = 'gray', linetype = '11'),
      panel.grid.minor = element_line(color = NA),
      axis.line.x = element_line(color="black", size = .25),
      axis.line.y = element_line(color="black", size = .25),
      legend.key = element_rect(color=NA, fill=alpha('white', .5)),
      legend.position = c(0.01, 0.01), 
      legend.justification = c(0, 0),
      legend.background = element_rect(fill=alpha('white', 0.4), size=0)
    )
  )
# A2. Plot wLP
g_int_wLP_2 <- ggplot(df, aes(x = wLP, y = int_dif_A3A1, col = grp)) +  
  geom_point(alpha=.7) + scale_color_manual(values=gcolors) +
  geom_smooth(method=lm, se=FALSE, fullrange=TRUE) +
  # scale_x_continuous(expand=c(0,0), limits=c(-10,10)) +
  # scale_y_continuous(expand=c(0,0), limits=c(-10,10)) +
  # coord_cartesian(xlim=c(-4,5), ylim=c(-10,10)) +
  labs(
    x = bquote(italic(w[LP])~(italic(z)-score)),
    col = 'Group') +
  theme_get() + theme(
    legend.position = "none",
    axis.title.y = element_blank()
  )

plotA <- arrangeGrob(g_int_wPC_2, g_int_wLP_2, ncol=2)
# plot(plotA)
ggsave(
  'intpref_A3A1.png',
  plot = plotA,
  path = '/Users/alexten/Desktop',
  width = 7,
  height = 3,
  units = 'in',
  dpi = 100
)