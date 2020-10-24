library(tidyverse)
setwd("~/Projects/MonsterStudy/data/raw")

dft <- rbind(tbl_df(read.csv('ig_extra.csv')), tbl_df(read.csv('eg_extra.csv'))) %>%
  select(c(age, gender, race))

median(dft$age, na.rm = TRUE)
mean(dft$age, na.rm= TRUE)
range(dft$age, na.rm = TRUE)

group_by(dft, race) %>% count()
