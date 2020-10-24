library(reshape2)
library(dplyr)
library(tidyr)

ntm <- tbl_df(read.csv('data/ntm_data_freeplay.csv')) %>% 
  dplyr::select(ntm,sid) %>% 
  dplyr::distinct() %>% mutate(sid=factor(sid))

dft <- read.csv('data/post_play_long.csv') %>% 
  tbl_df  %>% mutate(sid=factor(sid)) %>% 
  dplyr::inner_join(ntm, by='sid') %>% 
  dplyr::mutate_at(vars(sid,grp,ntm,tid), funs(factor)) %>% 
  dplyr::filter(ntm!=0) %>% 
  dplyr::select(sid,grp,ntm,tid,nlrn:nlrn2)
rm(ntm)

dft <- tidyr::gather(dft, item, rating, -c(sid,grp,ntm,tid)) %>% 
  dplyr::arrange(sid)

dft$grp <- recode(dft$grp, '0' = 'IG', '1' = 'EG')
dft$ntm <- recode(dft$ntm, '1' = 'NAM-1', '2' = 'NAM-2', '3' = 'NAM-3')
dft$tid <- recode(dft$tid, 
                  '1' = 'A1', 
                  '2' = 'A2',
                  '3' = 'A3',
                  '4' = 'A4')
dft$item <- factor(dft$item, levels=)

resTab <- tbl_df(data.frame(
  'grp'=numeric(0),
  'ntm'=numeric(0),
  'tid'=numeric(0),
  'grp.ntm'=numeric(0),
  'grp.tid'=numeric(0),
  'tid.ntm'=numeric(0),
  'grp.tid.ntm'=numeric(0)
  ))

coefs <- c('grp','ntm','tid','grp.ntm','grp.tid','tid.ntm','grp.tid.ntm')
items <- c('nint', 'ntime', 'nlrn2', 'nlrn', 'ncomp', 'nprog', 'nrule')
output <- matrix(ncol=length(coefs), nrow=length(items))
for (i in c(1:length(items))) {
  subdft <- dft[dft$item==items[i], ]
  linmod <- aov(rating~grp*tid*ntm+Error(sid), data=subdft)
  lmsummary <- summary(linmod)
  bw <- round(lmsummary[[1]][[1]]$'Pr(>F)'[1:3], 3)
  wi <- round(lmsummary[[2]][[1]]$'Pr(>F)'[1:4], 3)
  newRow <- c(bw, wi)[c(1,2,4,3,5,6,7)]
  output[i, ] = newRow
}

output <- as.data.frame(output, row.names=items)
colnames(output) <- coefs
write.table(output, '/Users/alexten/Desktop/output.csv', sep=',')
