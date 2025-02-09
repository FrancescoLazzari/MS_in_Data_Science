

# Stock per sectors function ----------------------------------------------

get.sectors <- function(sp500){
  
  # Extraction of all the unique sectors of the GICS
  sectors <- unique(sp500$SEC.filings)
  
  # Extraction of the companies symbols grouped by sector type
  symbols.by.sectors <- split(sp500$Tickers, sp500$SEC.filings)
  
  # Calculation of the number of the stocks into each sector
  table <- sapply(symbols.by.sectors, length)
  
  # Return the object table
  return(as.list(table))
}

# Stock data extraction function ------------------------------------------

stock.data <- function(stock){
  
  data <- get.hist.quote(instrument=stock, start="2020-01-01", end="2023-12-31",
                         quote= "Close", provider="yahoo", drop=FALSE, quiet = TRUE)$Close
  return(data)
}

# Cleaning function -------------------------------------------------------

stocks.filter <- function(symb){
  
  # Initialization of the vector that will contain the 
  # index of the symbols to remove because they contain a dot
  index_to_delete <- c()
  
  # Initialization of the vector that will contain the
  # symbols of the stocks to remove
  stocks_to_delete <- c()
  
  # Filter by dot (the stock symbols can't contain a dot)
  for (i in 1:length(symb)){
    
    if (grepl('\\.',symb[[i]])){
      
      index_to_delete <- c(index_to_delete, i)
      stocks_to_delete <- c(stocks_to_delete, symb[[i]])
    }
  }
  
  # Remove the symbols with dot from the input vector
  symb <- symb[-index_to_delete]
  
  # Filter by index (the stocks can't have different index (dates))
  AAPL_index <- index( get.hist.quote(instrument='AAPL', start="2020-01-01", end="2023-12-31",
                                 quote= c("Close"), provider="yahoo", drop=FALSE, quiet = TRUE) )
  
  for (i in 1:length(symb)){
    
    index_to_check <- index (get.hist.quote(instrument=symb[[i]], start="2020-01-01", end="2023-12-31",
                                            quote= c("Close"), provider="yahoo", drop=FALSE, quiet = TRUE) )
    
    if (identical(index_to_check, AAPL_index) == FALSE){
      stocks_to_delete <- c(stocks_to_delete, symb[[i]])
    }
  }
  
  # Return the vector that contains all the symbols to delete
  return(stocks_to_delete)
}


# Stock matrix function ---------------------------------------------------

stocks.matrix <- function(dimension, SP500){
  
  # Set seed for riproducybility
  set.seed(123)
  
  # Extraction of all the unique sectors of the GICS
  sectors <- names(dimension)
  
  # Vector with the number of samples for each sector
  samples <- as.numeric(dimension)
  
  # Total number of stocks
  p <- sum(samples)
  
  # Extraction of the companies symbols grouped by sector type
  symbols.by.sectors <- split(SP500$Tickers, SP500$SEC.filings)
  
  
  # Initialization of the matrix X 
  # To determine how many rows this matrix should have, we used the number of days 
  #   that the APPLE stock has been in the market 
  # We chose this stock as a benchmark because we assumed that, due to the company's 
  #   importance, it would not have missing data.
  date.index <- index( get.hist.quote(instrument='AAPL', start="2020-01-01", end="2023-12-31",
                                      quote= c("Close"), provider="yahoo", drop=FALSE, quiet = TRUE) )
  n.days = length(date.index)
  
  X <- matrix(NA, nrow = n.days, ncol = 0)
  
  # Sampling of the companies from each sectors
  for (i in 1:length(dimension)){
    
    # If the i-th sector has 0 samples we move to the next iteration
    if (samples[i] == 0) { next }

    # Extraction of the sample companies form the i-th sector
    temp.stocks <- sample(symbols.by.sectors[[ sectors[i] ]], samples[i] )
    
    new.column <- matrix(NA, n.days, samples[i])
    # Extraction of the closing price of the sample companies
    new.column <- sapply( temp.stocks, stock.data)
    
    # Add the data of the samples companies to the matrix X
    X = cbind(X, new.column)
  }
  
  # Rename the index of the matrix X with the day of the stock market
  rownames(X) <- as.character(date.index)
  
  # Replace each element of matrix X with the logarithm of the ratio between 
  # the closing price of that day and the closing price of the previous day
  X = log(X[2:nrow(X), ] / X[1:(nrow(X) - 1), ])
  
  # Of course, with this transformation, we lose the first row of the data since 
  # it is not possible to perform the previous transformation
  
  return(X)
}


# Find tau function -------------------------------------------------------

find.tau <- function(X, percentile){
  
  # Extract all the absolute value of the correlation coefficient
  # of every combination of stocks
  corr.list <- combn(ncol(X), 2, function(k) abs(cor(X[, k[1]], X[, k[2]])) )
  
  # Find tau which is equal to the 'percentile' of the correlation distribution
  tau <- quantile(corr.list, probs=percentile)
  
  return(tau)
  
}


# Correlation matrix ------------------------------------------------------

corr.matrix <- function(X, alpha, percentile){
  
  # Extraction of all the possible combinations of the p stocks
  combinations <- combn(colnames(X),2)
  
  # Number of stocks in the matrix
  p <- dim(X)[2]
  
  # Initialization of the correlation matrix by setting zero in each position
  c.matrix <- matrix(0, nrow = p, ncol = p, dimnames = list(colnames(X), colnames(X)) )
  
  # Extraction of the tau threshold through its dedicated function
  tau <- find.tau(X, percentile)
  
  # Creation of the tau-interval
  threshold <- c(-tau, tau)
  
  # Iterating through each unique combination of the p-stocks
  for (i in 1:dim(combinations)[2]){
    
    # Extraction of the vector containing the i-th combination
    pair <- combinations[ , i]
    
    # Extracting the p-value from cor.test to see if, for the given combination of stocks, 
    # their correlation is significantly different from 0
    p.value <- cor.test( X[ , pair[1] ], X[ , pair[2] ], method = "pearson", conf.level = 1-(alpha/p))$p.value
    
    # Since H0: rho=0, we want the p-value to be less than the threshold alpha/p (Bonferroni correction for Multiplicity Tests)
    if(p.value < alpha/p){
      
      # If we reach this step, we still cannot claim that the correlation is different from 0 because 
      # We still need to verify the condition given by the intersection between the confidence interval and the interval [-tau, +tau]
      
      # Extraction of the lower and upper bounds of the confidence interval
      ci <- cor.test( X[ , pair[1] ], X[ , pair[2] ], method = "pearson", conf.level = 1-(alpha/p))$conf.int
      
      # Check if the intersection of the two intervals is an empty set
      if (ci[2] < threshold[1] | ci[1] > threshold[2]){
        
        # If we reach this point of the function, then the pair of nodes meets all the conditions to claim 
        # that their correlation is different from 0
        
        # Therefore, we calculate the correlation value between the two stocks and insert it into the two 
        # symmetric positions of the matrix
        c.matrix[ pair[1], pair[2] ] <- c.matrix[ pair[2], pair[1] ] <- cor( X[ , pair[1] ] , X[ , pair[2] ] )
      }
    
    # If the p-value of cor.test is greater than alpha/p, it means that the available data show us a statistically 
    # significant correlation equal to 0, so no further checks are needed, and we let the correlation between those 
    # two stocks remain 0 (default value) and we move to the next pair of stock to evaluate
    } else { next }
  }
  
  # It returns the final correlation matrix, which will have values different from 0 (rounded to the second decimal place) 
  # only if all the conditions are met
  return(round(c.matrix,2))
}


# Edge Analysis -----------------------------------------------------------

edge_analysis <- function(g, sectors){
  # Extract the edges as a data frame
  edges_df <- as.data.frame(as.edgelist(g))
  
  # Count the number of edges
  num_edges <- nrow(edges_df)
  
  # Initialize counters for inter-sector and intra-sector edges
  inter_sector_edges <- 0
  intra_sector_edges <- 0
  
  # Loop through each edge
  for (i in 1:num_edges) {
    # Extract the sectors of the nodes connected by the edge
    node1_sector <- sectors[edges_df[i, 1]]
    node2_sector <- sectors[edges_df[i, 2]]
    
    # Check if the sectors are different
    if (node1_sector != node2_sector) {
      inter_sector_edges <- inter_sector_edges + 1
    } else {
      intra_sector_edges <- intra_sector_edges + 1
    }
  }

  # Create a data frame for the results
  results_df <- data.frame(
    "Parameter" = c("Number of Edges", "Inter-Sector Edges", "Intra-Sector Edges"),
    "Count" = c(num_edges, inter_sector_edges, intra_sector_edges)
  )
  return(results_df)
}


# Bootstrap Shapley value -------------------------------------------------

bootstrap_shapley <- function(data, indices, omega) {
  
  # Sampling with replacement for bootstrap
  sampled_data <- data[indices, ]
  
  # Covariance matrix
  cov.matrix <- cov(sampled_data)
  
  # Initialization of the vector containing the Shapley value of each of the p-stocks
  shapley_val <- rep(NaN, dim(sampled_data)[2])
  
  # Iteration through individual stocks
  for (i in 1:dim(sampled_data)[2]){
    
    # Expected value of the i-th stock
    exp.val <- mean(sampled_data[, i])
    
    # Sum of the covariances between the i-th stock and the other (p-1) stocks
    cov.sum <- sum(cov.matrix[, i])
    
    # Calculation of the Shapley value
    shapley_val[i] <- exp.val - (omega * cov.sum)
  }
  return(shapley_val)
}
