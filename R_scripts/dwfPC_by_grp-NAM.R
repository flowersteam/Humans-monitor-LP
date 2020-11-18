library(dplyr)
library(emmeans)

# Load and clean up data
dft <- tbl_df(read.csv('data/learning_data.csv'))
dft <- dplyr::filter(dft, nam != 0)
dft$group <- factor(dft$group)
dft$nam <- factor(dft$nam)

# Get NAM proportions by group
dft$nam <- factor(dft$nam)
(dplyr::group_by(dft, group, nam) %>% 
  dplyr::summarize(n=n()) %>% 
  dplyr::mutate(freq = n / sum(n)))

# Get dwfPC stats in each group
(dft %>% 
    dplyr::group_by(group) %>% 
    dplyr::summarize(M_dwfpc = mean(dwfpc), SD_dwfpc = sd(dwfpc))
)

# Fit a linear model of \ dwfPC ~ GROUP x NAM \
linmod <- lm(dwfpc ~ group * nam, dft)
anova(linmod)
emmeans(linmod, pairwise ~ group | nam) # Perform posthoc comparisons

# Fit a quadratic model of final weighted performance
# \* dwfPC ~ dwiPC + GROUP + SC^2 *\
dft <- dft %>% mutate(sc_lep_z = (sc_lep - mean(sc_lep))/sd(sc_lep))
dft <- dft %>% mutate(sc_lep_z2 = sc_lep_z^2)
quadmod <- lm(dwfpc ~ dwipc + group + sc_lep_z + sc_lep_z2, dft)
summary(quadmod)

# Fit a linear model of SC
# \* SC ~ GROUP x NAM *\
linmod2 <- lm(sc_lep ~ group * nam, dft)
summary(linmod2)
emmeans(linmod2, pairwise ~ group | nam) # Perform posthoc comparisons
