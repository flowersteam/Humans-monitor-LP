library(dplyr)
library(emmeans)

# Load and clean up data
dft <- tbl_df(read.csv('data/ntm_dwfpc.csv'))
dft <- dplyr::filter(dft, ntm != 0)
dft$grp <- factor(dft$grp)
dft$ntm <- factor(dft$ntm)

# Fit a linear model of dwfPC
linmod <- lm(post ~ grp * ntm, dft)
anova(linmod)
emmeans(linmod, pairwise ~ grp | ntm) # Perform posthoc comparisons

# Fit a quadratic model of SC
quadmod <- lm(post ~ pre + grp + poly(sc_lep, 2), dft)
summary(quadmod)

# Fit a linear model of SC
linmod2 <- lm(sc_lep ~ grp * ntm, dft)
summary(linmod2)
emmeans(linmod2, pairwise ~ grp | ntm) # Perform posthoc comparisons
