# Homework 1
# Authors #
# Cavicchioli Michael
# Landi Claudia
# Sani Vittorio

#install.packages("mvtnorm")
library(mvtnorm)

# Generazione dei dati e restituzione dei valori dei coefficienti
generate_data_and_get_coefficients <- function(i, n, sd){
  
  # Set del seed
  set.seed(111+i)
  
  # Scenario con solo x1
  x = rmvnorm(n, c(0, 0), sd)
  x1 = x[,1]
  x2 = x[,2]
  e = rnorm(n,0, 1)
  
  y = b0 + b1*x1 + b2*x2 + e
  
  model_x1 = lm(y ~ x1)
  summary(model_x1)
  
  # Scenario con sia x1 che x2
  y1 = b0 + b1*x1 + b2*x2 + e
  model_x1_x2 = lm(y1 ~ x1+x2)
  summary(model_x1_x2)
  
  return (list(coef(model_x1)[2], coef(model_x1_x2)[2]))
}

# Plot di un istogramma
histogram <- function(x, xlab, col1, col2, xlim, main){
  {hist(x,  probability=TRUE, xlab = xlab, 
        ylab = "Frequenze", col = col1,
        breaks = seq(min(x), max(x), length.out = 20), 
        xlim = xlim, ylim = c(0, 4),
        main = main
  )
    lines(density(x), col = col2, lwd = 2)}
}

# Plot di una curva
plt <- function(x, y, xlim, main){
  {plot(density(x), xlim = xlim, ylim = c(0,4.0), main = main)
    lines(density(y), col = "darkred")}
}

# Coefficienti
b0 = 2
b1 = -0.5
b2 = 1

# Matrici di varianza e covarianza
sd1 <- matrix(c(1, 0.5, 0.5, 1), nrow = 2, ncol = 2)
sd2 <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2)

n = 100  # Campione
nsim = 2000  # Numero simulazioni

# Inizializzazione delle variabili
b1_x1_s1 = c()      #b1 con solo x1 scenario 1
b1_x1_x2_s1 = c()   #b1 con sia x1 che x2 scenario 1
b1_x1_s2 = c()      #b1 con solo x1 scenario 2
b1_x1_x2_s2 = c()   #b1 con sia x1 che x2 scenario 2

# Simulazioni Monte Carlo
for (i in 1:nsim) {
  
  # Scenario 1
  results <- generate_data_and_get_coefficients(i, n, sd1)
  b1_x1_s1[i] <- results[[1]]
  b1_x1_x2_s1[i] <- results[[2]]
  
  # Scenario 2
  results <- generate_data_and_get_coefficients(i, n, sd2)
  b1_x1_s2[i] <- results[[1]]
  b1_x1_x2_s2[i] <- results[[2]]
}

# Grafici
# Layout per gli istogrammi
par(mfrow = c(1, 2))

# Istogrammi scenario 1
histogram(b1_x1_s1, "b1 - x1", "blue", "darkred", c(-1.1, 1.1), "Val. di b1 con solo x1 come predittore")
histogram(b1_x1_x2_s1, "b1 - x1, x2", "red", "darkblue", c(-1.1, 0.2), "Val. di b1 con x1 e x2 come predittori")

# Istogrammi scenario 2
histogram(b1_x1_s2, "b1 - x1", "blue", "darkred", range(b1_x1_s2), "Val. di b1 con solo x1 come predittore")
histogram(b1_x1_x2_s2, "b1 - x1, x2", "red", "darkblue", range(b1_x1_x2_s2), "Val. di b1 con x1 e x2 come predittori")

# Confronto scenario 1 e 2
plt(b1_x1_s1, b1_x1_x2_s1, c(-1.1, 0.5), "Scenario 1")
plt(b1_x1_s2, b1_x1_x2_s2, range(b1_x1_s2), "Scenario 2")

# Differenza tra scenari
plt(b1_x1_s1, b1_x1_s2, c(-1.1, 0.5), "Solo x1")
plt(b1_x1_x2_s1, b1_x1_x2_s2, range(b1_x1_x2_s1), "Sia x1, che x2")
