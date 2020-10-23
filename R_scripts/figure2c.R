library(dplyr)
library(plyr)
library(ggplot2)

# Load and clean up data
dft <- tbl_df(read.csv('data/ntm_dwfpc.csv'))
dft <- dplyr::filter(dft, ntm != 0)
dft <- dplyr::select(dft, c(sid, grp, post))

mu <- ddply(dft, "grp", summarise, grp.mean=mean(post))
head(mu)

sdev <- ddply(dft, "grp", summarise, grp.sd=sd(post))
head(sdev)

# Define looks
dft$grp <- recode(dft$grp, '0' = 'IG', '1' = 'EG')
gcolors = c('#fc4f30', '#008fd5')
# gcolors <- c(rep('#fc4f30', 'IG'), rep('#008fd5', 'EG'))

g <- ggplot(dft, aes(x = post, y = stat(count) / sum(count), fill=grp)) +  
  geom_histogram(alpha=.6, position='identity', bins = 15) +
  scale_color_manual(values=gcolors) + scale_fill_manual(values=gcolors) +
  labs(
    y = 'Relative frequency',
    x = 'dwfPC') +
  theme_set(
    theme(
      panel.background = element_blank(),
      panel.grid.major = element_line(color = 'gray', linetype = '11'),
      panel.grid.minor = element_line(color = NA),
      axis.text = element_text(size=15),
      axis.title = element_text(size=18),
      axis.line.x = element_line(color="black", size = .25),
      axis.line.y = element_line(color="black", size = .25),
      legend.key = element_rect(color=NA, fill=alpha('white', .5)),
      legend.position = c(0.01, 0.99),
      legend.justification = c(0, 1),
      legend.text = element_text(size=15),
      legend.title = element_blank(),
      legend.background = element_rect(fill=alpha('white', 0.4), size=0)
    )
  )


plot(g)

ggsave(
  '2c.png',
  plot = g,
  path = '/Users/alexten/Projects/MonsterStudy/Paper/png',
  width = 5,
  height = 4,
  units = 'in',
  dpi = 300
)
