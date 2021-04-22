library(kernlab)
library('latex2exp')

gauss.kernel <- function(x, y = 0, sig = 0.2){
  return(exp(-1/(2*sig**2) * abs(x - y) ** 2))
}

h <- function(k, z1, z2){
  return(k(z1[1], z2[1]) + k(z1[2], z2[2]) - k(z1[1], z2[2]) - k(z1[2], z2[1]))
}

MMD <- function(k, z){
  n <- length(x)
  summe <- 0
  for(i in 1:n){
    for(j in (1:n)[-i]){
      summe <- summe + h(k, z[i,], z[j,])
    }
  }
  return(1/(n*(n-1)) * summe)
}


m <- 1000
n <- 100

vers1 <- replicate(m, sqrt(n) * MMD(gauss.kernel, matrix(data = c(runif(n), runif(n) + 1),ncol = 2)))
hist(vers1, main = TeX("$\\widehat{MMD}^{2}$ für $P^{X} \\neq P^{Y}$"), xlab = TeX("$\\widehat{MMD}^{2}$"), ylab = "freq", freq = F)

vers2 <- replicate(m, sqrt(n) * MMD(gauss.kernel, matrix(data = c(runif(n), runif(n)),ncol = 2)))
hist(vers2, main = TeX("$\\widehat{MMD}^{2}$ für $P^{X} = P^{Y}$"), xlab = TeX("$\\widehat{MMD}^{2}$"), ylab = "freq", freq = F)

