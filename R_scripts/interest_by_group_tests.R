library(tidyverse)
library(emmeans) 

# Load nam dataset
nam_df <- tbl_df(read.csv('data/nam_data.csv')) %>% 
  dplyr::select(nam,sid) %>% 
  dplyr::distinct() %>% mutate(sid=factor(sid), nam=factor(nam))

# Load self-reports dataset and merge with NAM data
df <- tbl_df(read.csv('data/combined_extra.csv')) %>% 
  dplyr::mutate(sid=factor(sid), group=factor(group)) %>% 
  dplyr::inner_join(nam_df, by='sid')

# Select data for one questionnaire item
df <- dplyr::filter(df, item=='int', nam!=0) %>% rename(int = rating_norm)

# Load and hoin relative time on activities data
df2 <- tbl_df(read.csv('data/model_data.csv')) %>% 
  dplyr::select(sid, trial, relt1, relt2, relt3, relt4) %>%
  dplyr::filter(trial==250) %>%
  dplyr::rename(A1=relt1, A2=relt2, A3=relt3, A4=relt4) %>%
  tidyr::gather(activity, relt, A1:A4) %>% 
  dplyr::mutate(sid=factor(sid), activity=factor(activity))

df <- dplyr::left_join(df, df2, by=c('sid'='sid', 'activity'='activity'))
df <- within(df, nam <- relevel(nam, ref = '2'))

# Fit linear model
lm_ <- lm(int ~ relt * group, data=df)
summary(lm_)

# # Perform post hoc analysis
posthoc <- emmeans(lm_, ~ time*nam | group)
print(pairs(posthoc))
