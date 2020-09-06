#install.packages("coin")
library(coin)
#install.packages("multcomp")
library(multcomp)

#install.packages("PMCMR")
library(PMCMR)

x <- rep(c("gaussian", "knn", "parzen"), 300)

x1 <- rep(c(1:300), each=3)

resp <- c(0.715, 0.745, 1.   ,
       0.775, 0.785, 1.   ,
       0.71 , 0.73 , 1.   ,
       0.73 , 0.72 , 1.   ,
       0.77 , 0.725, 1.   ,
       0.72 , 0.725, 1.   ,
       0.705, 0.72 , 1.   ,
       0.685, 0.78 , 1.   ,
       0.76 , 0.765, 1.   ,
       0.665, 0.745, 1.   ,
       0.705, 0.75 , 1.   ,
       0.71 , 0.75 , 1.   ,
       0.75 , 0.805, 1.   ,
       0.73 , 0.785, 1.   ,
       0.745, 0.75 , 1.   ,
       0.755, 0.81 , 1.   ,
       0.7  , 0.71 , 1.   ,
       0.725, 0.78 , 1.   ,
       0.76 , 0.795, 1.   ,
       0.69 , 0.715, 1.   ,
       0.725, 0.75 , 1.   ,
       0.75 , 0.74 , 1.   ,
       0.69 , 0.77 , 1.   ,
       0.735, 0.725, 1.   ,
       0.72 , 0.77 , 1.   ,
       0.76 , 0.795, 1.   ,
       0.68 , 0.725, 1.   ,
       0.705, 0.745, 1.   ,
       0.705, 0.745, 1.   ,
       0.76 , 0.7  , 1.   ,
       0.7  , 0.695, 1.   ,
       0.74 , 0.78 , 1.   ,
       0.685, 0.695, 1.   ,
       0.695, 0.725, 1.   ,
       0.71 , 0.76 , 1.   ,
       0.745, 0.77 , 1.   ,
       0.73 , 0.72 , 1.   ,
       0.715, 0.72 , 1.   ,
       0.755, 0.755, 1.   ,
       0.725, 0.75 , 1.   ,
       0.765, 0.785, 1.   ,
       0.73 , 0.72 , 1.   ,
       0.745, 0.77 , 1.   ,
       0.765, 0.795, 1.   ,
       0.685, 0.735, 1.   ,
       0.705, 0.765, 1.   ,
       0.72 , 0.7  , 1.   ,
       0.73 , 0.715, 1.   ,
       0.72 , 0.715, 1.   ,
       0.69 , 0.72 , 1.   ,
       0.71 , 0.715, 1.   ,
       0.725, 0.79 , 1.   ,
       0.78 , 0.735, 1.   ,
       0.765, 0.69 , 1.   ,
       0.695, 0.745, 1.   ,
       0.715, 0.78 , 1.   ,
       0.715, 0.755, 1.   ,
       0.71 , 0.755, 1.   ,
       0.695, 0.78 , 1.   ,
       0.73 , 0.73 , 1.   ,
       0.665, 0.72 , 1.   ,
       0.75 , 0.72 , 1.   ,
       0.695, 0.735, 1.   ,
       0.705, 0.78 , 1.   ,
       0.72 , 0.725, 1.   ,
       0.745, 0.82 , 1.   ,
       0.755, 0.755, 1.   ,
       0.715, 0.775, 1.   ,
       0.735, 0.765, 1.   ,
       0.745, 0.75 , 1.   ,
       0.73 , 0.76 , 1.   ,
       0.69 , 0.705, 1.   ,
       0.73 , 0.76 , 1.   ,
       0.735, 0.76 , 1.   ,
       0.695, 0.72 , 1.   ,
       0.735, 0.76 , 1.   ,
       0.73 , 0.78 , 1.   ,
       0.72 , 0.75 , 1.   ,
       0.725, 0.77 , 1.   ,
       0.725, 0.79 , 1.   ,
       0.725, 0.73 , 1.   ,
       0.765, 0.715, 1.   ,
       0.76 , 0.775, 1.   ,
       0.675, 0.72 , 1.   ,
       0.715, 0.76 , 1.   ,
       0.7  , 0.73 , 1.   ,
       0.695, 0.69 , 1.   ,
       0.745, 0.755, 1.   ,
       0.72 , 0.765, 1.   ,
       0.725, 0.76 , 1.   ,
       0.725, 0.68 , 1.   ,
       0.72 , 0.72 , 1.   ,
       0.695, 0.75 , 1.   ,
       0.725, 0.785, 1.   ,
       0.74 , 0.75 , 1.   ,
       0.76 , 0.765, 1.   ,
       0.685, 0.73 , 1.   ,
       0.775, 0.78 , 1.   ,
       0.695, 0.71 , 1.   ,
       0.73 , 0.8  , 1.   ,
       0.745, 0.76 , 1.   ,
       0.74 , 0.74 , 1.   ,
       0.73 , 0.775, 1.   ,
       0.71 , 0.745, 1.   ,
       0.72 , 0.77 , 1.   ,
       0.715, 0.725, 1.   ,
       0.695, 0.74 , 1.   ,
       0.7  , 0.775, 1.   ,
       0.755, 0.77 , 1.   ,
       0.755, 0.735, 1.   ,
       0.755, 0.73 , 1.   ,
       0.675, 0.725, 1.   ,
       0.675, 0.685, 1.   ,
       0.755, 0.79 , 1.   ,
       0.73 , 0.75 , 1.   ,
       0.72 , 0.72 , 1.   ,
       0.745, 0.745, 1.   ,
       0.745, 0.755, 1.   ,
       0.705, 0.745, 1.   ,
       0.735, 0.76 , 1.   ,
       0.72 , 0.75 , 1.   ,
       0.765, 0.735, 1.   ,
       0.71 , 0.745, 1.   ,
       0.755, 0.71 , 1.   ,
       0.705, 0.71 , 1.   ,
       0.71 , 0.76 , 1.   ,
       0.73 , 0.765, 1.   ,
       0.695, 0.74 , 1.   ,
       0.72 , 0.745, 1.   ,
       0.705, 0.72 , 1.   ,
       0.715, 0.765, 1.   ,
       0.735, 0.77 , 1.   ,
       0.735, 0.725, 1.   ,
       0.735, 0.745, 1.   ,
       0.755, 0.775, 1.   ,
       0.655, 0.725, 1.   ,
       0.69 , 0.705, 1.   ,
       0.74 , 0.73 , 1.   ,
       0.735, 0.78 , 1.   ,
       0.685, 0.775, 1.   ,
       0.775, 0.745, 1.   ,
       0.695, 0.735, 1.   ,
       0.72 , 0.76 , 1.   ,
       0.71 , 0.75 , 1.   ,
       0.74 , 0.795, 1.   ,
       0.71 , 0.725, 1.   ,
       0.73 , 0.77 , 1.   ,
       0.715, 0.705, 1.   ,
       0.74 , 0.79 , 1.   ,
       0.695, 0.715, 1.   ,
       0.68 , 0.73 , 1.   ,
       0.73 , 0.725, 1.   ,
       0.735, 0.76 , 1.   ,
       0.63 , 0.7  , 1.   ,
       0.72 , 0.78 , 1.   ,
       0.715, 0.73 , 1.   ,
       0.76 , 0.785, 1.   ,
       0.755, 0.785, 1.   ,
       0.78 , 0.785, 1.   ,
       0.73 , 0.715, 1.   ,
       0.725, 0.765, 1.   ,
       0.695, 0.74 , 1.   ,
       0.695, 0.76 , 1.   ,
       0.785, 0.775, 1.   ,
       0.71 , 0.73 , 1.   ,
       0.73 , 0.73 , 1.   ,
       0.71 , 0.72 , 1.   ,
       0.72 , 0.77 , 1.   ,
       0.735, 0.765, 1.   ,
       0.695, 0.715, 1.   ,
       0.715, 0.76 , 1.   ,
       0.71 , 0.75 , 1.   ,
       0.735, 0.785, 1.   ,
       0.67 , 0.7  , 1.   ,
       0.72 , 0.79 , 1.   ,
       0.73 , 0.76 , 1.   ,
       0.715, 0.705, 1.   ,
       0.775, 0.78 , 1.   ,
       0.735, 0.72 , 1.   ,
       0.715, 0.775, 1.   ,
       0.705, 0.775, 1.   ,
       0.73 , 0.74 , 1.   ,
       0.7  , 0.745, 1.   ,
       0.71 , 0.78 , 1.   ,
       0.73 , 0.75 , 1.   ,
       0.745, 0.72 , 1.   ,
       0.7  , 0.745, 1.   ,
       0.735, 0.745, 1.   ,
       0.745, 0.71 , 1.   ,
       0.72 , 0.705, 1.   ,
       0.665, 0.745, 1.   ,
       0.695, 0.705, 1.   ,
       0.725, 0.7  , 1.   ,
       0.725, 0.735, 1.   ,
       0.675, 0.76 , 1.   ,
       0.77 , 0.79 , 1.   ,
       0.79 , 0.78 , 1.   ,
       0.71 , 0.735, 1.   ,
       0.725, 0.755, 1.   ,
       0.725, 0.755, 1.   ,
       0.685, 0.735, 1.   ,
       0.67 , 0.715, 1.   ,
       0.695, 0.77 , 1.   ,
       0.73 , 0.785, 1.   ,
       0.765, 0.77 , 1.   ,
       0.68 , 0.715, 1.   ,
       0.76 , 0.735, 1.   ,
       0.735, 0.79 , 1.   ,
       0.74 , 0.765, 1.   ,
       0.77 , 0.785, 1.   ,
       0.725, 0.805, 1.   ,
       0.72 , 0.76 , 1.   ,
       0.75 , 0.775, 1.   ,
       0.68 , 0.745, 1.   ,
       0.71 , 0.72 , 1.   ,
       0.735, 0.755, 1.   ,
       0.74 , 0.76 , 1.   ,
       0.73 , 0.78 , 1.   ,
       0.69 , 0.74 , 1.   ,
       0.73 , 0.75 , 1.   ,
       0.705, 0.73 , 1.   ,
       0.785, 0.74 , 1.   ,
       0.735, 0.755, 1.   ,
       0.73 , 0.795, 1.   ,
       0.705, 0.695, 1.   ,
       0.74 , 0.73 , 1.   ,
       0.705, 0.74 , 1.   ,
       0.71 , 0.74 , 1.   ,
       0.725, 0.78 , 1.   ,
       0.705, 0.77 , 1.   ,
       0.755, 0.745, 1.   ,
       0.71 , 0.77 , 1.   ,
       0.665, 0.72 , 1.   ,
       0.69 , 0.705, 1.   ,
       0.77 , 0.77 , 1.   ,
       0.7  , 0.75 , 1.   ,
       0.71 , 0.74 , 1.   ,
       0.785, 0.8  , 1.   ,
       0.645, 0.7  , 1.   ,
       0.8  , 0.79 , 1.   ,
       0.76 , 0.815, 1.   ,
       0.68 , 0.755, 1.   ,
       0.755, 0.74 , 1.   ,
       0.755, 0.735, 1.   ,
       0.705, 0.765, 1.   ,
       0.675, 0.675, 1.   ,
       0.715, 0.73 , 1.   ,
       0.69 , 0.715, 1.   ,
       0.735, 0.755, 1.   ,
       0.715, 0.735, 1.   ,
       0.715, 0.675, 1.   ,
       0.735, 0.755, 1.   ,
       0.76 , 0.79 , 1.   ,
       0.685, 0.745, 1.   ,
       0.74 , 0.74 , 1.   ,
       0.695, 0.725, 1.   ,
       0.715, 0.715, 1.   ,
       0.725, 0.75 , 1.   ,
       0.73 , 0.755, 1.   ,
       0.73 , 0.76 , 1.   ,
       0.71 , 0.74 , 1.   ,
       0.695, 0.725, 1.   ,
       0.74 , 0.72 , 1.   ,
       0.655, 0.775, 1.   ,
       0.75 , 0.77 , 1.   ,
       0.71 , 0.75 , 1.   ,
       0.74 , 0.78 , 1.   ,
       0.715, 0.745, 1.   ,
       0.755, 0.75 , 1.   ,
       0.73 , 0.715, 1.   ,
       0.725, 0.735, 1.   ,
       0.655, 0.715, 1.   ,
       0.695, 0.755, 1.   ,
       0.745, 0.76 , 1.   ,
       0.75 , 0.76 , 1.   ,
       0.725, 0.725, 1.   ,
       0.755, 0.755, 1.   ,
       0.72 , 0.75 , 1.   ,
       0.7  , 0.76 , 1.   ,
       0.76 , 0.8  , 1.   ,
       0.725, 0.72 , 1.   ,
       0.68 , 0.75 , 1.   ,
       0.71 , 0.735, 1.   ,
       0.73 , 0.74 , 1.   ,
       0.715, 0.765, 1.   ,
       0.76 , 0.77 , 1.   ,
       0.725, 0.76 , 1.   ,
       0.71 , 0.735, 1.   ,
       0.73 , 0.755, 1.   ,
       0.745, 0.785, 1.   ,
       0.7  , 0.765, 1.   ,
       0.72 , 0.77 , 1.   ,
       0.745, 0.735, 1.   ,
       0.735, 0.76 , 1.   ,
       0.7  , 0.725, 1.   ,
       0.715, 0.685, 1.   ,
       0.68 , 0.745, 1.   ,
       0.725, 0.785, 1.   ,
       0.715, 0.76 , 1.   ,
       0.785, 0.8  , 1.   )


friedman.test(resp, groups=x, blocks=x1)
posthoc.friedman.nemenyi.test(resp, groups=x, blocks=x1)


