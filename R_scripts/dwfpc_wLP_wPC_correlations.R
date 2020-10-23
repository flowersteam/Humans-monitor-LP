library(dplyr)
library(plyr)
library(emmeans)

# Load and clean up data
dft <- tbl_df(read.csv('data/ntm_dwfpc.csv'))
dft <- dplyr::filter(dft, ntm != 0)
dft <- dplyr::select(dft, c(sid, post, pre))

# Get coefficients data
df <- tbl_df(read.csv('data/coef_data.csv')) %>% 
  dplyr::rename(wPC=pc_coef, wLP=rlp_coef) %>% 
  dplyr::mutate(A4minusA3 = -A3minusA4)

mu <- ddply(df, "grp", summarise, grp.mean=mean(wLP))
head(mu)

sdev <- ddply(df, "grp", summarise, grp.sd=sd(wLP))
head(sdev)

# Join datasets and optionally filter poorly fitted models
df <- dplyr::inner_join(df, dft, by='sid')
rm(dft)

# # Define looks
# df$grp <- recode(df$grp, '0' = 'IG', '1' = 'EG')
# gcolors = c('#008fd5', '#fc4f30')
# 
# g <- ggplot(df, aes(x = post, y = wPC, col = grp)) +  
#   geom_point(alpha=.3) + scale_color_manual(values=gcolors) + 
#   geom_smooth(method=lm, se=FALSE, fullrange = TRUE) +
#   labs(
#     y = bquote(italic(w[PC])),
#     x = 'dwfPC',
#     col = 'Group') +
#   theme_set(
#     theme(
#       panel.background = element_blank(),
#       panel.grid.major = element_line(color = 'gray', linetype = '11'),
#       panel.grid.minor = element_line(color = NA),
#       axis.line.x = element_line(color="black", size = .25),
#       axis.line.y = element_line(color="black", size = .25),
#       legend.key = element_rect(color=NA, fill=alpha('white', .5)),
#       legend.position = c(0.01, 0.01),
#       legend.justification = c(0, 0),
#       legend.background = element_rect(fill=alpha('white', 0.4), size=0)
#     )
#   )
# 
# g2 <- ggplot(df, aes(x = post, y = wLP, col = grp)) +  
#   geom_point(alpha=.3) + scale_color_manual(values=gcolors) + 
#   geom_smooth(method=lm, se=FALSE, fullrange = TRUE) +
#   labs(
#     y = bquote(italic(w[LP])),
#     x = 'dwfPC',
#     col = 'Group') + theme_get() +
#     theme(
#       legend.position = "none"
#     )
# 
# plots <- arrangeGrob(g, g2, nrow=2, ncol=1)
# 
# ggsave(
#   'dwfPC_correlations.png',
#   plot = plots,
#   path = '/Users/alexten/Desktop',
#   width = 3,
#   height = 6,
#   units = 'in',
#   dpi = 100
# )

# Linear models
lm1 <- lm(wPC ~ grp, data=df)
summary(aov(lm1))

lm2 <- lm(wLP ~ grp, data=df)
summary(aov(lm2))
