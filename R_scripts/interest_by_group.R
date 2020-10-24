library(tidyverse)
library(ggplot2)

# Load ntm dataset
ntm <- tbl_df(read.csv('data/ntm_data_freeplay.csv')) %>% 
  dplyr::select(ntm,sid) %>% 
  dplyr::distinct() %>% mutate(sid=factor(sid))

# Get ratings dataset
dft <- tbl_df(read.csv('data/post_play_long.csv')) %>% 
  dplyr::select(sid,grp,tid,int) %>%
  dplyr::mutate(sid=factor(sid)) %>% 
  dplyr::inner_join(ntm, by='sid')
rm(ntm)

dft <- filter(dft, tid>=4, ntm>0)
dft$grp <- factor(dft$grp)
dft$ntm <- factor(dft$ntm)
linmod <- lm(int ~ grp*ntm, data=dft)
summary(linmod)

dft <- dft %>%
  dplyr::group_by(grp, ntm) %>%
  dplyr::summarize_at(vars(int), funs(mean = mean, sem = sd(.)/sqrt(n()))) %>%
  dplyr::ungroup() %>% 
  dplyr::arrange(grp,ntm)

dft$grp <- factor(dft$grp)
dft$ntm <- factor(dft$ntm)
dft$grp <- recode(dft$grp, '0' = 'IG', '1' = 'EG')

g <- dft %>% 
  ggplot(
    aes(x = ntm, 
        y = mean, 
        col = grp,
        group = grp, 
        ymin = mean-sem,
        ymax = mean+sem)) +  
  geom_line() +
  geom_errorbar(
    width=.2,
    position=position_dodge(0.05)) +
  labs(
    y = 'Average interest (raw)',
    x = 'NAM',
    col = '') +
  scale_color_manual(values=c('#008fd5', '#fc4f30')) +
  theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(color = 'gray', linetype = '11'),
    panel.grid.minor = element_line(color = NA),
    axis.line.x = element_line(color="black", size = .25),
    axis.line.y = element_line(color="black", size = .25),
    legend.key = element_rect(color=NA, fill=alpha('white', .5)),
    legend.position = c(0.5, 0.01),
    legend.justification = c(.5, 0),
    legend.background = element_blank()
  )

plot(g)
ggsave(
  'int_ratings.png',
  plot = g,
  path = '/Users/alexten/Desktop',
  width = 3,
  height = 2,
  units = 'in',
  dpi = 100
)