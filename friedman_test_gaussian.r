#install.packages("coin")
library(coin)
#install.packages("multcomp")
library(multcomp)

#install.packages("PMCMR")
library(PMCMR)

x <- rep(c("soma","fou","kar"), 300)

x1 <- rep(c(1:300), each=3)

resp <- c(0.715,0.575,0.775,0.775,0.59,0.79,0.71,0.56,0.75,0.73,0.645,0.73,0.77,0.605,0.8,0.72,0.59,0.705,0.705,0.56,0.76,0.685,0.585,0.75,0.76,0.655,0.79,0.665,0.6,0.74,0.705,0.56,0.75,0.71,0.58,0.765,0.75,0.585,0.8,0.73,0.62,0.75,0.745,0.585,0.775,0.755,0.6,0.745,0.7,0.585,0.73,0.725,0.575,0.805,0.76,0.675,0.77,0.69,0.565,0.725,0.725,0.535,0.755,0.75,0.6,0.79,0.69,0.615,0.75,0.735,0.6,0.74,0.72,0.62,0.79,0.76,0.585,0.805,0.68,0.535,0.735,0.705,0.61,0.765,0.705,0.59,0.715,0.76,0.66,0.73,0.7,0.59,0.73,0.74,0.63,0.78,0.685,0.56,0.715,0.695,0.56,0.735,0.71,0.54,0.775,0.745,0.63,0.755,0.73,0.57,0.795,0.715,0.595,0.755,0.755,0.625,0.76,0.725,0.585,0.765,0.765,0.635,0.825,0.73,0.585,0.73,0.745,0.605,0.805,0.765,0.605,0.79,0.685,0.575,0.745,0.705,0.59,0.775,0.72,0.61,0.725,0.73,0.575,0.74,0.72,0.59,0.74,0.69,0.58,0.73,0.71,0.53,0.78,0.725,0.605,0.745,0.78,0.665,0.81,0.765,0.6,0.74,0.695,0.57,0.75,0.715,0.6,0.745,0.715,0.62,0.775,0.71,0.575,0.745,0.695,0.615,0.73,0.73,0.56,0.77,0.665,0.545,0.725,0.75,0.605,0.78,0.695,0.575,0.715,0.705,0.575,0.74,0.72,0.57,0.75,0.745,0.635,0.785,0.755,0.64,0.795,0.715,0.6,0.76,0.735,0.58,0.78,0.745,0.585,0.78,0.73,0.55,0.775,0.69,0.57,0.745,0.73,0.615,0.785,0.735,0.59,0.76,0.695,0.6,0.735,0.735,0.655,0.76,0.73,0.615,0.755,0.72,0.595,0.745,0.725,0.58,0.77,0.725,0.585,0.775,0.725,0.6,0.75,0.765,0.595,0.815,0.76,0.63,0.725,0.675,0.585,0.71,0.715,0.57,0.8,0.7,0.59,0.785,0.695,0.57,0.7,0.745,0.605,0.755,0.72,0.61,0.78,0.725,0.565,0.76,0.725,0.58,0.785,0.72,0.625,0.735,0.695,0.58,0.785,0.725,0.61,0.745,0.74,0.615,0.785,0.76,0.6,0.795,0.685,0.58,0.7,0.775,0.58,0.83,0.695,0.595,0.71,0.73,0.61,0.775,0.745,0.62,0.76,0.74,0.58,0.77,0.73,0.62,0.76,0.71,0.59,0.735,0.72,0.565,0.79,0.715,0.57,0.76,0.695,0.605,0.72,0.7,0.565,0.745,0.755,0.645,0.78,0.755,0.59,0.805,0.755,0.57,0.755,0.675,0.565,0.735,0.675,0.525,0.725,0.755,0.6,0.805,0.73,0.565,0.76,0.72,0.58,0.745,0.745,0.67,0.79,0.745,0.615,0.755,0.705,0.615,0.725,0.735,0.595,0.755,0.72,0.58,0.765,0.765,0.61,0.785,0.71,0.61,0.73,0.755,0.585,0.75,0.705,0.645,0.735,0.71,0.585,0.755,0.73,0.57,0.78,0.695,0.58,0.715,0.72,0.605,0.79,0.705,0.59,0.76,0.715,0.575,0.775,0.735,0.605,0.76,0.735,0.63,0.725,0.735,0.65,0.755,0.755,0.59,0.76,0.655,0.575,0.72,0.69,0.605,0.74,0.74,0.55,0.795,0.735,0.58,0.8,0.685,0.57,0.765,0.775,0.61,0.765,0.695,0.6,0.725,0.72,0.595,0.76,0.71,0.555,0.775,0.74,0.64,0.785,0.71,0.585,0.73,0.73,0.615,0.785,0.715,0.61,0.77,0.74,0.57,0.77,0.695,0.565,0.74,0.68,0.59,0.675,0.73,0.605,0.77,0.735,0.545,0.775,0.63,0.535,0.675,0.72,0.61,0.755,0.715,0.615,0.765,0.76,0.575,0.815,0.755,0.59,0.775,0.78,0.605,0.815,0.73,0.595,0.76,0.725,0.59,0.755,0.695,0.63,0.725,0.695,0.63,0.73,0.785,0.6,0.79,0.71,0.58,0.77,0.73,0.565,0.775,0.71,0.595,0.73,0.72,0.59,0.775,0.735,0.565,0.8,0.695,0.59,0.74,0.715,0.565,0.78,0.71,0.535,0.755,0.735,0.62,0.735,0.67,0.585,0.735,0.72,0.585,0.745,0.73,0.59,0.78,0.715,0.645,0.755,0.775,0.65,0.79,0.735,0.58,0.765,0.715,0.595,0.75,0.705,0.59,0.77,0.73,0.555,0.785,0.7,0.645,0.725,0.71,0.575,0.775,0.73,0.595,0.8,0.745,0.655,0.745,0.7,0.53,0.725,0.735,0.6,0.795,0.745,0.615,0.73,0.72,0.605,0.73,0.665,0.57,0.675,0.695,0.545,0.765,0.725,0.585,0.785,0.725,0.58,0.75,0.675,0.605,0.76,0.77,0.63,0.785,0.79,0.61,0.835,0.71,0.575,0.8,0.725,0.62,0.72,0.725,0.6,0.755,0.685,0.54,0.76,0.67,0.53,0.7,0.695,0.58,0.77,0.73,0.58,0.78,0.765,0.615,0.79,0.68,0.56,0.69,0.76,0.655,0.755,0.735,0.585,0.775,0.74,0.665,0.78,0.77,0.62,0.815,0.725,0.59,0.81,0.72,0.59,0.73,0.75,0.61,0.835,0.68,0.585,0.73,0.71,0.575,0.745,0.735,0.6,0.74,0.74,0.61,0.75,0.73,0.605,0.73,0.69,0.59,0.735,0.73,0.58,0.79,0.705,0.585,0.765,0.785,0.635,0.75,0.735,0.56,0.76,0.73,0.595,0.76,0.705,0.6,0.735,0.74,0.62,0.76,0.705,0.53,0.775,0.71,0.575,0.79,0.725,0.63,0.765,0.705,0.565,0.72,0.755,0.575,0.775,0.71,0.595,0.745,0.665,0.56,0.705,0.69,0.595,0.695,0.77,0.64,0.79,0.7,0.56,0.77,0.71,0.585,0.775,0.785,0.62,0.815,0.645,0.54,0.715,0.8,0.655,0.845,0.76,0.665,0.785,0.68,0.58,0.76,0.755,0.625,0.79,0.755,0.595,0.78,0.705,0.585,0.765,0.675,0.58,0.71,0.715,0.59,0.75,0.69,0.56,0.73,0.735,0.57,0.775,0.715,0.585,0.735,0.715,0.53,0.78,0.735,0.62,0.8,0.76,0.61,0.77,0.685,0.55,0.75,0.74,0.585,0.78,0.695,0.6,0.725,0.715,0.58,0.76,0.725,0.575,0.735,0.73,0.59,0.74,0.73,0.65,0.74,0.71,0.59,0.755,0.695,0.6,0.765,0.74,0.64,0.77,0.655,0.5,0.705,0.75,0.585,0.825,0.71,0.58,0.755,0.74,0.64,0.76,0.715,0.595,0.755,0.755,0.55,0.81,0.73,0.635,0.735,0.725,0.59,0.735,0.655,0.55,0.705,0.695,0.56,0.74,0.745,0.595,0.8,0.75,0.63,0.785,0.725,0.575,0.755,0.755,0.625,0.79,0.72,0.58,0.76,0.7,0.62,0.785,0.76,0.6,0.765,0.725,0.575,0.745,0.68,0.565,0.74,0.71,0.59,0.77,0.73,0.58,0.74,0.715,0.59,0.745,0.76,0.605,0.78,0.725,0.61,0.77,0.71,0.575,0.745,0.73,0.625,0.78,0.745,0.61,0.8,0.7,0.62,0.73,0.72,0.55,0.75,0.745,0.585,0.77,0.735,0.605,0.75,0.7,0.575,0.74,0.715,0.57,0.785,0.68,0.585,0.745,0.725,0.575,0.75,0.715,0.62,0.8,0.785,0.645,0.79)


friedman.test(resp, groups=x, blocks=x1)
posthoc.friedman.nemenyi.test(resp, groups=x, blocks=x1)


