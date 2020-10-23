library(tidyverse)

df <- read.csv('data/trials_data2.csv')
df <- tbl_df(df)
df <- df %>%
  dplyr::select(sid, grp, trial, loc_pc1, loc_pc2, loc_pc3)  %>%
  dplyr::mutate_at(vars(sid, grp), funs(factor)) %>%
  dplyr::filter(trial>60) %>%
  gather(tid, pc, loc_pc1:loc_pc3) %>% arrange(sid)

df$tid <- recode(df$tid, 
                 'loc_pc1' = 'A1', 
                 'loc_pc2' = 'A2',
                 'loc_pc3' = 'A3')
df$tid <- factor(df$tid)
df['mastered'] = df$pc >= 13/15
df <- df[, c('sid','grp','tid','pc','mastered')]
df <- group_by(df, grp, sid, tid)
df <- summarize(df, mt_present=any(mastered))
df <- ungroup(df) %>% group_by(grp, sid)
df <- summarize(df, nam=sum(mt_present)) %>% ungroup
df <- group_by(df, grp, nam) %>% summarize(n=n()) %>% mutate(freq = n / sum(n))

filter(df, nam>=2) %>% summarize(cump=sum(freq))

# Did dwfPC final performance differ by NAM and GRP?
df <- tbl_df(read.csv('data/ntm_dwfpc.csv'))
df <- df[df$ntm>0, c('sid','grp','ntm','post')]

# Mixed ANOVA
AOV_results <- aov(post ~ grp * ntm, df)
summary(AOV_results)

