library(coin)
library(multcomp)
library(PMCMR)

x <- rep(c({labels}), 300)

x1 <- rep(c(1:300), each={size})

resp <- c({matrix})


friedman.test(resp, groups=x, blocks=x1)
posthoc.friedman.nemenyi.test(resp, groups=x, blocks=x1)


