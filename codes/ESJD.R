ESJD <- function(Data) {
  # Calculate dimensions
  N_Data <- nrow(Data)
  dim_theta <- ncol(Data)
  
  # Calculate differences between consecutive rows
  Delta <- Data[-1, ] - Data[-N_Data, ]
  N_Delta <- N_Data - 1
  
  # Vectorized calculation of the covariance matrix equivalent
  re <- t(Delta) %*% Delta / N_Delta
  
  # Calculate the determinant and normalize
  re <- det(re)^(1 / dim_theta)
  
  return(re)
}
