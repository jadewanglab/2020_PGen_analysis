library(tidyverse)
library(broom)

# read in data from Table S2, Nicolas, Mader, Dervyn et al -------
dat = read_csv('TableS2_Nicolas_et_al.csv') %>%
  filter(grepl("^BSU", Locus_tag)) # keep only locus_tags beginning with BSU.

rtp_end = dat$EndV3[which(dat$Name=='rtp')] # locate end of rtp gene to later assign codirectional and headon status

# assign codirectional and head-on depending on strand and location relative to rtp
dat = dat %>%
  mutate(headon = as.factor(ifelse(StartV3 > rtp_end,
                         ifelse(Strand == 1, 1, 0),
                         ifelse(Strand == 1, 0, 1))))

# tidy and z-score the data ----
convert_to_robustz <- function(vec){
  finite = is.finite(vec)
  this_median = median(vec[finite])
  mad = median(abs(vec[finite] - this_median))
  output = (vec - this_median)/(mad*1.4826)
  output
}

dat_long = dat %>%
  dplyr::select(Name, Locus_tag, headon, `LBexp_1_hyb25350202`:`MG+150_3_hyb14630502`) %>%
  gather(key="condition", value="log2signal", `LBexp_1_hyb25350202`:`MG+150_3_hyb14630502`) %>%
  mutate(signal=2**log2signal) %>%
  separate(condition, into=c("condition", "replicate", "hybID", "stuff"), sep="_") %>%
  mutate(sample = paste(condition, replicate, sep="_")) %>%
  separate(condition, into=c("condition", "other"), sep="/") %>%
  mutate(condition = ifelse(condition=="MG+t5", "MG+5", condition)) %>%
  group_by(condition, replicate) %>%
  mutate(z_score = convert_to_robustz(log2signal)) %>%
  ungroup()

# save tidied data to csv file
write_csv(dat_long, path='data_long.csv')