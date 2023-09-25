#########################################################
# Course: Machine Learning for BI 1 
# Lecture 1: Fundamentals and KNN alg.
# Copyright: Ana Alina Tudoran
# Aarhus University, Fall 2023
#########################################################

# KNN regression - Application

    # Modeling packages 
    library(rsample)   # for resampling procedures
    library(caret)     # for resampling and model training # a meta-engine

    # Ames housing data
    ames <- AmesHousing::make_ames() # data is available here
    dim(ames)
    str(ames$Sale_Price)

    # Data partitioning (we apply stratified sampling here) 
    set.seed(123)
    split <- initial_split(ames, prop = 0.7, 
                           strata = "Sale_Price")
    ames_train  <- training(split)
    ames_test   <- testing(split)
    
    hist(ames$Sale_Price)
    hist(ames_train$Sale_Price)
    hist(ames_test$Sale_Price)
    # Comment: Notice the same distribution of DV in the new datasets
    

    # Specify re-sampling strategy: k-fold cross-validation
    cv <- trainControl(
      method = "repeatedcv", 
      number = 10, 
      repeats = 1
    )
    
    # Create a grid of values for the hyperparameter k
    hyper_grid <- expand.grid(k = seq(2, 25, by = 1))
    hep <- expand.grid(k = seq(2, 25, by = 1), p = seq(2,25, by=1))

    # Tune a knn model using grid search (here we use caret package)
    knn_fit <- train(
      Sale_Price ~ ., 
      data = ames_train, 
      method = "knn", 
      trControl = cv, 
      tuneGrid = hyper_grid,
      metric = "RMSE"
    )
    knn_fit
  # Comments
    # We see that the best model for this run (with seed 123) is associated with 
    # k = 7, which resulted in a cv-RMSE of 43607.26 
    knn_fit$resample$RMSE
    # Here RMSE was used to select the optimal model using the smallest value. 
    # We can use another metric (e.g. MSE) for model selection. Try to change the 
    # code and display the the result.

  # Extra code    
  # Plot cv-error
  ggplot(knn_fit)

  # Test error (on test data)
  pred = predict(knn_fit, newdata=ames_test)
  pred
  test_error = sqrt(mean((ames_test$Sale_Price - pred)^2))
  test_error
  # Comments  
    # [1] test error = 42406.73 is close to the cv-error (43607.26). 
    # On average the algorithm overestimate and underestimate the value of houses 
    # by approx. 42.000 - 43.000$. Can we do something to improve the performance? 
    # We will see next lecture how feature engineering may improve the algorithm 
    # performance
  
  # If we want to get the training error for k = 7
   pred_train = predict(knn_fit, newdata=ames_train)
   pred_train
   train_error_train = sqrt(mean((ames_train$Sale_Price - pred_train)^2))
   train_error_train
  # Comments
    # As expected it is lower than test error, but not significantly lower. 
    # => There are no signs of overfitting. 

  # Final comments
    # This was an example of fitting a knn regression, with k-fold cv, in caret. 
    # Later, we return with an example of knn classification.

