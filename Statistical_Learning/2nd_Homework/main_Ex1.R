
# Statistical Learning - HW2 Part 1

# G06 - Solo Lacrime 🫠
# Francesco Lazzari, Camilla Brigandì, Matteo Pazzini, Paolo Meli, Riccardo Violano


# Import Packages ---------------------------------------------------------

library(xgboost)
library(dplyr)
library(Matrix)
library(caret)
library(readr)
library(zoo)
library(parallel)  # yes, let's use all the CPU available :)

# Set the working directory ----------------------------------------------
directory <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(directory)

# Import Dataset ----------------------------------------------------------

# Maybe you want to change the paths, maybe...
train <- read_csv("Data/train.csv")
test <- read_csv("Data/test.csv")

set.seed(543)  # 123 was too boring

# lets remove our 10 lucky rows 🎉
i = sample(1:nrow(train), 10)
train = train[-i, ]

# Preprocessing and dimensionality reduction on train ---------------------

# one hot encoding of the 'day' column
days <- data.frame(model.matrix(~ day - 1, data = train))
names(days) <- c('Afternoon', 'Evening', 'Morning', 'Night')

# one hot encoding of the 'month' column
months <-  model.matrix(~ month - 1, data = train)

# creation of an empty dataframe to store the one hot encoded months grouped by seasons
seasons <- data.frame(Spring = integer(nrow(train)), 
                      Summer = integer(nrow(train)), 
                      Autumn = integer(nrow(train)), 
                      Winter = integer(nrow(train)))

# assign for each row the correct season
seasons$Spring <- rowSums(months[ , c(1, 8, 9)] ) > 0
seasons$Summer <- rowSums(months[ , c(2, 6, 7)] ) > 0
seasons$Autumn <- rowSums(months[ , c(10, 11, 12)] ) > 0
seasons$Winter <- rowSums(months[ , c(3, 4, 5)] ) > 0

# convert the boolean values into dummies
seasons <- data.frame(lapply(seasons, as.integer))

# extraction of the columns with the observations for each quantitative variable
speed <- train[ , 6:35]
cadence <- train[ , 36:65]
altitude <- train[ , 66:95]
names(speed) = as.character(c(1:30))
names(cadence) = as.character(c(1:30))
names(altitude) = as.character(c(1:30))

# initialization of two variable for store the '∆ altitude' upwards and downwards 
upwards <- numeric(nrow(train))
downwards <- numeric(nrow(train))

# difference between each observation
diffs <- t(apply(altitude, 1, diff))

# cumulative difference upwards and downwards
upwards <- rowSums(diffs * (diffs > 0))
downwards <- rowSums(-diffs * (diffs < 0))

# substitute the 'NA' values with the last observed value
fill_na <- function(x) {
  # if the NA is at the beginning substitute it with the first value not NA
  x <- na.locf(x, na.rm = FALSE)
  # if the NA is at the center/end substitute it with the last value non NA
  x <- na.locf(x, fromLast = TRUE, na.rm = FALSE)
  return(x)
}

# apply the custom function to the cadence observations
cadence <- t(apply(cadence, 1, fill_na))

# the remaining rows that have NA in all positions 
# we will treat them later sooo stay tuned ...
sum(is.na(cadence))/30

#fill the only missing value in speed with the value right after
speed[ which(is.na(speed[ , 1])), 1] = speed[ which(is.na(speed[ , 1])) , 2]


# dimensionality reduction through some statistics for each quantitative features + the one hot encoded qualitative features  --> yes, a very complex solution 🤯
new_train <- cbind(train$id,
                  as.numeric(as.factor(train$y))-1, # numeric target zones
                  train$from_start,
                  days,
                  seasons,
                  apply(speed, 1, mean, na.rm = T),           # µ(speed)
                  apply(speed, 1, max,na.rm = T),             # max(speed)
                  apply(speed, 1, sd,na.rm = T),              # sd(speed)
                  (((speed$'30'-speed$'1')*1000/3600)/60),    # µ(acceleration)
                  apply(cadence, 1, mean, na.rm = T),         # µ(cadence)
                  apply(cadence, 1, max, na.rm = T),          # max(cadence)
                  apply(cadence, 1, sd, na.rm = T),           # sd(cadence)
                  apply(altitude, 1, mean),                   # µ(altitude)
                  apply(altitude, 1, max),                    # max(altitude)
                  apply(altitude, 1, sd),                     # sd(altitude)
                  upwards, 
                  downwards
                  )

# rename some columns with a readable name
names(new_train)[1:3] <-c('original_id', 'y', 'from_start')

names(new_train)[12:21] <- c('mu_s', 'max_s', 'sd_s', 'avg_acceleration', 
                             'mu_c','max_c','sd_c', 
                             'mu_a','max_a','sd_a')

for (i in 1:nrow(new_train)){
  
  for (j in 16:18){
    
    if (is.na(new_train[i, j]) | new_train[i,j] == -Inf){
      
      if (new_train$max_s[i] < 1){
        new_train[i,j] = min(new_train[ ,j], na.rm = T)
      }
      else {
        new_train[i,j] = mean(new_train[ ,j], na.rm = T)
      }
    }
  }
}

new_train$max_c <- ifelse(new_train$max_c == -Inf, new_train$mu_c, new_train$max_c)


# Normalization of all the columns
new_train <- new_train %>%
  mutate(across(c(from_start, mu_s, max_s, sd_s, avg_acceleration,
                  mu_c, max_c, sd_c, mu_a, max_a, sd_a, upwards, downwards),
                ~ scale(.)[, 1]))

# Preprocessing and dimensionality reduction on test ----------------------

# one hot encoding of the 'day' column
days <- data.frame(model.matrix(~ day - 1, data = test))
names(days) <- c('Afternoon', 'Evening', 'Morning', 'Night')

# one hot encoding of the 'month' column
months <-  model.matrix(~ month - 1, data = test)

# creation of an empty dataframe to store the one hot encoded months grouped by seasons
seasons <- data.frame(Spring = integer(nrow(test)), 
                      Summer = integer(nrow(test)), 
                      Autumn = integer(nrow(test)), 
                      Winter = integer(nrow(test)))

# assign for each row the correct season
seasons$Spring <- rowSums(months[ , c(1, 8, 9)] ) > 0
seasons$Summer <- rowSums(months[ , c(2, 6, 7)] ) > 0
seasons$Autumn <- rowSums(months[ , c(10, 11, 12)] ) > 0
seasons$Winter <- rowSums(months[ , c(3, 4, 5)] ) > 0

# convert the boolean values into dummies
seasons <- data.frame(lapply(seasons, as.integer))

# estraction of the columns with the observations for each quantitative variable
speed <- test[ , 5:34]
cadence <- test[ , 35:64]
altitude <- test[ , 65:94]
names(speed) = as.character(c(1:30))
names(cadence) = as.character(c(1:30))
names(altitude) = as.character(c(1:30))

# fill the only missing value in altitude and speed with the value right after
altitude[ which(is.na(apply(altitude, 1, mean))), 1] = altitude[ which(is.na(apply(altitude, 1, mean))),2] 
speed[ which(is.na(speed[,1])), 1] = speed[ which(is.na(speed[,1])), 2]



# initialization of two variable for store the '∆ altitude' upwards and downwards 
upwards <- numeric(nrow(test))
downwards <- numeric(nrow(test))

# difference between each observation
diffs <- t(apply(altitude, 1, diff))

# cumulative difference upwards and downwards
upwards <- rowSums(diffs * (diffs > 0))
downwards <- rowSums(-diffs * (diffs < 0))

# apply the custom function to the cadence observations
cadence <- t(apply(cadence, 1, fill_na))

# the remaining rows that have NA in all positions 
# we will treat them later sooo stay tuned ...
sum(is.na(cadence))/30

# dimensionality reduction throught some statistics for each quantitative features + the one hot encoded qualitative features  --> yes, a very complex solution 🤯
new_test <- cbind(test$id,
                  test$from_start,
                  days,
                  seasons,
                  apply(speed, 1, mean, na.rm = T),           # µ(speed)
                  apply(speed, 1, max,na.rm = T),             # max(speed)
                  apply(speed, 1, sd,na.rm = T),              # sd(speed)
                  (((speed$'30'-speed$'1')*1000/3600)/60),    # µ(acceleration)
                  apply(cadence, 1, mean, na.rm = T),         # µ(cadence)
                  apply(cadence, 1, max, na.rm = T),          # max(cadence)
                  apply(cadence, 1, sd, na.rm = T),           # sd(cadence)
                  apply(altitude, 1, mean),                   # µ(altitude)
                  apply(altitude, 1, max),                    # max(altitude)
                  apply(altitude, 1, sd),                     # sd(altitude)
                  upwards, 
                  downwards
                  )

# rename some columns with a readable name
names(new_test)[1:2] <-c('original_id', 'from_start')

names(new_test)[11:20] <- c('mu_s', 'max_s', 'sd_s', 'avg_acceleration', 
                             'mu_c','max_c','sd_c', 
                             'mu_a','max_a','sd_a')

for (i in 1:nrow(new_test)){
  
  for (j in 15:17){
    
    if (is.na(new_test[i,j]) | new_test[i,j] == -Inf){
      
      if (new_test$max_s[i] < 1){
        new_test[i,j] = min(new_test[ ,j], na.rm = T)
      }
      else {
        new_test[i,j] = mean(new_test[ ,j], na.rm = T)
      }
    }
  }
}

new_test$max_c <- ifelse(new_test$max_c == -Inf, new_test$mu_c, new_test$max_c)

# Normalization of all the columns
new_test <- new_test %>%
  mutate(across(c(from_start, mu_s, max_s, sd_s, avg_acceleration,
                  mu_c, max_c, sd_c, mu_a, max_a, sd_a, upwards, downwards),
                ~ scale(.)[, 1]))



# Extreme Gradient Boosting 😮 ---------------------------------------------

# Extraction of the target labels
label_train <- new_train[ , 'y']

# copy of the new_train object
sparse_matrix <- new_train

# remove the not useful columns
sparse_matrix <- sparse_matrix %>% select(-original_id)
sparse_matrix <- sparse_matrix %>% select(-y)

# Creation of a DMatrix object
dtrain <- xgb.DMatrix(data = as.matrix(sparse_matrix), label =  label_train)


## Parameters Tuning ------------------------------------------------------

base_params <- list(
  objective = "multi:softmax",  # we are not interested into the prob on each class
  num_class = 6,
  eval_metric = "mlogloss",  
  nthread = detectCores()
)

# grid of parameters to evaluate through CV
# ⚠️: these are only the optimal values that we found for `submission_25`
grid <- expand.grid(
  max_depth = c(11),
  eta = 0.01,
  gamma = c(0.4),
  colsample_bytree = c(0.7),
  min_child_weight =  c(2),
  subsample = c(0.85)
)


# Function for the tuning that can be executed in parallel
tune_params <- function(i) {
  # yes, we are calling 2 global variables 'grid' & 'base_params' inside a function ♥
  params <- modifyList(base_params, list(
    max_depth = grid$max_depth[i],
    eta = grid$eta[i],
    gamma = grid$gamma[i],
    colsample_bytree = grid$colsample_bytree[i],
    min_child_weight = grid$min_child_weight[i],
    subsample = grid$subsample[i],
    lambda = grid$lambda[i],
    alpha = grid$alpha[i]
  ))
  
  cv_result <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 2000,
    nfold = 5,
    stratified = TRUE,
    verbose = FALSE,
    early_stopping_rounds = 100,
    maximize = FALSE
  )
  
  mean_mlogloss <- min(cv_result$evaluation_log$test_mlogloss_mean)
  list(mean_mlogloss = mean_mlogloss, params = params)
}

# yes, it would be a safer solution to not use all the cores but since we are brave (on kaggle) and not in local let's see what happens 🤗
results <- mclapply(1:nrow(grid), tune_params, mc.cores = 8)

# best parameters selection
best_result <- results[[which.min(sapply(results, function(res) res$mean_mlogloss))]]
best_params <- best_result$params


## Model training with validation -------------------------------------------

# in order to reduce the risk of overfitting we decided to do the final training only 
# on the 80% of the data, keeping the remaining 20% for validation purposes
indexv = sample(1:nrow(new_train), floor(0.2 * nrow(new_train)))
indext = c(1:nrow(new_train))[-indexv]

# this way the model will keep training on the training set till the results computed on 
# the validation set stop improving, making the model much more likely to perform better
# on datas outside the training set
best_model <- xgb.train(
  params = best_params,
  data = dtrain[as.integer(indext), ],
  watchlist = list(train = dtrain[as.integer(indext), ], val = dtrain[as.integer(indexv), ]),
  early_stopping_rounds = 100,
  nrounds = 10000,
  maximize = F
)

# let's go with the predictions 🚀

id_test <- new_test$original_id
test <- new_test %>% select(-original_id)

preds <- predict(best_model, newdata = as.matrix(test))


## Feature importance ------------------------------------------------------

# why? well, why not?

# Get feature importance
importance_matrix <- xgb.importance(model = best_model)

# Plot feature importance 
xgb.plot.importance(importance_matrix)


## Submit ----------------------------------------------------------------

# initialization of a matrix full of 0 --> yes, this comment was necessary
pred_matrix <- matrix(0, nrow = length(preds), ncol = 6)

# fill the previous matrix 
for (i in 1:length(preds)) {
  pred_matrix[i, preds[i]+1] <- 1
}

pred_matrix <- as.data.frame(pred_matrix)

# rename the columns
colnames(pred_matrix) <- c("Z0", "Z1", "Z2", "Z3", "Z4", "Z5")

# add the test 'id'
pred_matrix <- cbind(id_test, pred_matrix)
names(pred_matrix)[1] <- 'id'

# let's se the frequencies of the predicted classes
apply(pred_matrix, 2 , sum)

# save the submission
write.csv(pred_matrix, "submit_28.csv", row.names = FALSE)

