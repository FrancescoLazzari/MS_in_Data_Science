
find.threshold = function(N, p, q) {
  # Calculate the parameters of the approximate normal distribution of the honest studets
  mean_h0 = N * p
  dev_std_h0 = sqrt(N * p * (1 - p))  
  
  # Calculate the parameters of the approximate normal distribution of the liar studets
  mean_h1 = N * q
  dev_std_h1 = sqrt(N * q * (1 - q))
  
  # Find the intersection with the uniroot function
  intersection = uniroot(
    function(x) dnorm(x, mean = mean_h0, sd = dev_std_h0) - dnorm(x, mean = mean_h1, sd = dev_std_h1),
    interval = c(0, N)
  )
  
  # Round the root to the nearest integer
  threshold = round(intersection$root)
  
  return(threshold)
}


# Error Difference Matrix -------------------------------------------------

calculate_Error_Difference = function(N_max, alpha_UMP, q_val) {
  
  p = 0.5
  
  Error = matrix(rep(0, N_max * length(q_val) * length(alpha_UMP)), ncol = 7)
  
  Error_difference = matrix(rep(0, length(q_val) * length(alpha_UMP)), 
                            nrow = length(alpha_UMP), dimnames = list(as.character(alpha_UMP), NULL))
  
  for (k in seq_along(alpha_UMP)) {
    current_alpha_UMP = alpha_UMP[k]
    
    for (j in seq_along(q_val)) {
      q = q_val[j]
      
      for (i in 1:N_max) {
        t = find.threshold(i, p, q)
        
        Error[i, 1] = 1 - pbinom(t, i, p)
        Error[i, 2] = pbinom(t, i, q)
        Error[i, 3] = Error[i, 1] + Error[i, 2]
        Error[i, 4] = current_alpha_UMP
        Error[i, 5] = pbinom(qbinom(1 - current_alpha_UMP, i, p), i, q)
        Error[i, 6] = Error[i, 4] + Error[i, 5]
        Error[i, 7] = Error[i, 3] < Error[i, 6]
      }
      
      Error_difference[k, j] = round(sum(Error[, 7]) / N_max, 2)
    }
  }
  
  return(Error_difference)
}


# Score Function 1 --------------------------------------------------------

score.function1 = function(N, alpha.star, beta.star, q){
  
  p = 0.5
  
  # Find alpha an beta associated to the parameters N and q
  t = find.threshold(N, p, q)
  alpha = 1 - pbinom(t, N, p)
  beta = pbinom(t, N, q)
  
  if(alpha>alpha.star){
    
    # Calculate the distance between alpha and alpha.star
    D_alpha = (alpha-alpha.star)
    
  } else{ D_alpha = 0 }
  
  if(beta> beta.star){
    
    # Calculate the distance between beta and beta.star
    D_beta = (beta-beta.star)
  } else{ D_beta = 0 }
  
  # Calculate the mean between the two distances.
  Score = (1/2) * (D_alpha + D_beta)
  
  # Return the of the prvious score
  return(1-Score)
  
}

# Growth rate fuction -----------------------------------------------------

growth_rate <- function(data, step) {
  n = length(data)
  growths = rep(0, n)
  
  for (i in 1:n) {
    if (i > n-step) {
      growths[i] = NaN
    } else {
      growths[i] = (data[i + step] - data[i]) / data[i]
    }
  }
  
  return(growths)
}


# Find N substar ----------------------------------------------------------

find.N.substar = function(N_max, alpha.star, beta.star, q) {
  # Definitio of the range in which the score function is calculated
  N_values = 1:N_max  
  
  # Calculate the score function for all the N_values
  scores = sapply(N_values, function(N) score.function1(N, alpha.star, beta.star, q))
  
  # Apply the moving average with 10 terms to the score function
  smoothed_scores = stats::filter(scores, rep(1/10, 10), sides = 2)
  
  # Calculate all growth rates with step 5 on the smoothed score function
  rate_of_growth = growth_rate(smoothed_scores, 5)
  
  # Find N_sub_star comparing growth rates with a treshold=0.01
  N_sub_star <- N_values[which(rate_of_growth < 0.01 )[1]]
  
  return(N_sub_star)
}

# Calculate N substar -----------------------------------------------------

calculate_N_substar = function(N_max, alpha.star, beta.star, q_values) {
  results = matrix(0, nrow = 1, ncol = length(q_values))
  
  for (i in 1:length(q_values)) {
    q = q_values[i]
    N_star = find.N.substar(N_max, alpha.star, beta.star, q)[1]
    results[1,i] = N_star
  }
  
  return(results)
}


# Find N Star -------------------------------------------------------------

find.N.star = function(alpha.star, beta.star,q){
  Max_iterations = 1000
  
  for(i in 1:Max_iterations){
    if(score.function1(i,alpha.star, beta.star, q) == 1){
      return(i)
    }
  }
}



# Score Function 2 --------------------------------------------------------

score.function2= function( alpha.star, beta.star, q, Time){
  
  p=0.5
  # Find N_star using a specific function that returns 
  # the first value of the score.function.1 with a score equal to 1.
  N_star = find.N.star(alpha.star, beta.star, q)
  
  # Calculate the maximum number of tosses possible for half of the students 
  # within the established time constraint
  N_max = (Time*60)/(3*75) 
  
  N = seq(1,N_max, 1)
  Score_T= rep(0, N_max)
  
  for(i in 1:N_max){
    if(N_max <= N_star){
      # case 1 
      Score_T[i] = score.function1(i, alpha.star, beta.star, q)
      
    } else if( N_max > N_star){
      # case 2
      if(i<=N_star){
        Score_T[i] = score.function1(i, alpha.star, beta.star, q)
      } else{
        w = (N_max - (i-N_star))/N_max
        Score_T[i] = score.function1(i, alpha.star, beta.star, q) * w
      }
    }
  }
  return(Score_T)
}


# Exercise 2 --------------------------------------------------------------

# KERNEL ESTIMATOR DENSITY ------------------------------------------------



### Epanechnikov kernel density estimator

f_estimator <- function(x, sample, bandwidth) {
  N <- length(sample)
  
  #the epanechnikov kernel
  k <- function(u) {
    return((3/4) * (1 - u^2) * (abs(u) <= 1))
  }
  k_vectorized = Vectorize(k)
  
  f_hat <- rep(0, length(x))
  
  for (i in 1:N) {
    u <- (x - sample[i]) / bandwidth
    f_hat <- f_hat + k_vectorized(u) / (N * bandwidth)
  }
  
  return(f_hat)
}



### Cumulative distribution of KDE
I <- function(z){
  result <- ifelse(abs(z) < 1, (1/2) + (3/4) * (z - (z^3)/3),
                   ifelse(z < -1, 0, 1))
  
}


CDF_hat <- function(x,beta_data, bandwidth){
  N = length(beta_data)
  (1/N)*sum(I((x-beta_data)/bandwidth))
}



CDF_hat_vectorized <- Vectorize(CDF_hat,vectorize.args = "x")


# Quantile function of KDE
# Define the inverse of your CDF (quantile function)
my_quantile <- function(p, beta_data, h) {
  uniroot(function(x) CDF_hat_vectorized(x, beta_data, h) - p, interval = c(-2, 2))$root
}


my_quantile_vectorized = Vectorize(my_quantile,vectorize.args = "p")



compute_wasserstein_distance <- function(alpha,beta, sample, bandwith) {
  
  my_quantile <- function(p, beta_data = sample, h = bandwidth) {
    uniroot(function(x) CDF_hat_vectorized(x, beta_data, h) - p, interval = c(-2, 2))$root
  }
  
  my_quantile_vectorized = Vectorize(my_quantile,vectorize.args = "p")
  
  #Target: beta quantile function
  quantile_funct <- function(x) qbeta(x,alpha,beta)
  
  integrand <- function(z) abs(quantile_funct(z)-my_quantile_vectorized(z))
  integrand_vectorized = Vectorize(integrand)
  
  # Compute the p-Wasserstein distance
  wasserstein_distance <- integrate(integrand_vectorized,lower = 0, upper = 1, subdivisions = 10000)$value  #we limit the subdivision due to complexity
  
  return(wasserstein_distance)
}

