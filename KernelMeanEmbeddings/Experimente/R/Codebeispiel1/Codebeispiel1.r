library(kernlab)

?kmmd

x <- matrix(rnorm(200), 100) 
y <- matrix(runif(200), 100) 
kmmd(x, y)

x <- matrix(rnorm(200, 0, 1), 100)
y <- matrix(rnorm(200, 0, 1.5), 100) 
kmmd(x, y)

x <- matrix(rnorm(200), 100)
y <- matrix(rnorm(200), 100)
kmmd(x, y)
