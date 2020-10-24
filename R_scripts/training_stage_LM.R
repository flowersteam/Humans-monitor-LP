library(dplyr)       # common data transformation functions
library(emmeans)     # `emmeans` function for performing pst hoc analyses on fitted models

# Open data into a tibble and select what is needed
df <- read.csv('data/trials_data2.csv')
df <- tbl_df(df)
df <- df %>%
  dplyr::select(sid, grp, trial, pc1, pc2, pc3, pc4)  %>%
  dplyr::mutate_at(vars(sid, grp), funs(factor)) %>%
  dplyr::filter(trial==60) %>%
  gather(tid, pc, pc1:pc4) %>% arrange(sid)

# Recode and reformat
df$tid <- recode(df$tid,
                 'pc1' = 'A1',
                 'pc2' = 'A2',
                 'pc3' = 'A3',
                 'pc4' = 'A4')
df$tid <- factor(df$tid)
df <- df[, c('sid','grp','tid','pc')]

# Fit a linear model
LM_model <- lm(pc ~ grp * tid, data=df, contrasts=list(grp=contr.treatment(2), tid=contr.treatment(4)))

# Perform mixed ANOVA
AOV_results <- aov(pc ~ grp * tid + Error(sid), df)
summary(AOV_results)

# Perform post hoc analysis
posthoc <- emmeans(LM_model, 'tid', adjust = "tukey")
print(pairs(posthoc))