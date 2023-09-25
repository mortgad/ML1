#########################################################
# Course: Machine Learning for BI 1 
# Lecture 2: Feature engineering application
# Copyright: Ana Alina Tudoran
# Aarhus University, Fall 2023
#########################################################

# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics
library(visdat)   # for additional visualizations
# Feature engineering packages
library(caret)    # for various ML tasks
library(recipes)  # for feature engineering tasks

#__________
# 3.1. Data "ames" has property sales information
#__________
#   DV: Sale_Price (i.e., $195,000, $215,000)
#   Objective: use property attributes to predict the sale price of a home
#   Details: ?AmesHousing::ames_raw
#   Install.packages("AmesHousing", lib="/Library/Frameworks/R.framework/Versions/4.0/Resources/library")

# Get data
  ames <- AmesHousing::make_ames()

# Stratified sampling with the rsample package
  set.seed(123)
  library(rsample)
  split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
  ames_train  <- training(split)
  ames_test   <- testing(split)


#__________  
# 3.2 Target engineering 
#__________
    attach(mtcars)
    par(mfrow=c(1,3))
    hist(ames_train$Sale_Price, breaks = 20, 
         col = "red", border = "red", 
         ylim = c(0, 800))
   
    # log if all value are positive
    transformed_response <- log(ames_train$Sale_Price)
    
    # Box-Cox transformation (lambda=0 is equivalent to log(x))
    library(forecast)
    transformed_response_BC <- forecast::BoxCox(ames_train$Sale_Price, 
                                                lambda="auto") 
    
    hist(transformed_response, breaks = 20, 
         col = "lightgreen", border = "lightgreen",
         ylim = c(0, 800) )
    hist(transformed_response_BC, breaks = 20, 
         col = "lightblue", border = "lightblue", 
         ylim = c(0, 800))
    
    
    # log transformation using a blueprint (recipe)
    # As an example, we create ames_recipe using caret as: 
      ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
        step_log(all_outcomes())
      ames_recipe # This will not return the actual log transformed values 
      # but, rather, a blueprint to be applied later.
    
#__________
# 3.3 Dealing with missing data
#__________
    # *NOTE: this is a hypothetical situation for ames_raw data; 
    # in most of applications in HOM, ames data set is used, which contains no 
    # missing values

    # (1) Always ensure the missing data are coded correctly (NA)
      # If missing data has a particular label code in the original file
      # (such as, unknown, 999, *, etc.), recode that label to missing value
        # data[data =="unknown"] <- NA 
        # data[data == 999] <- NA 
        # data[data == "*"] <- NA 
      
    # (2) How many missing?  
    # If we use the original ames_raw data (via AmesHousing::ames_raw), 
    # dim(AmesHousing::ames_raw)
      sum(is.na(AmesHousing::ames_raw)) # 13,997 missing values 

    # (3) Look at patterns of missingness
        # simple plot
          library(DataExplorer)
          plot_missing(AmesHousing::ames_raw) 
       
        # advanced plot
        library(ggplot2) 
        AmesHousing::ames_raw %>%
          is.na() %>%
          reshape2::melt() %>%
          ggplot(aes(Var2, Var1, fill=value)) + 
          geom_raster() + 
          coord_flip() +
          scale_y_continuous(NULL, expand = c(0, 0)) +
          scale_fill_grey(name = "", 
                          labels = c("Present", 
                                     "Missing")) +
          xlab("Observation") +
          theme(axis.text.y  = element_text(size = 4))
        # from the plot:
          # The majority of missing values occur in the variables Alley, Fireplace 
          # Qual, Pool QC, Fence, and Misc Feature. Due to their high frequency 
          # of missingness, these variables would likely need to be removed prior
          # to statistical analysis 
          # Also, we observe missing values appear to occur within the same 
          # observations across all "garage" variables.
          AmesHousing::ames_raw %>% 
            filter(is.na(`Garage Type`)) %>% 
            select(`Garage Type`, `Garage Cars`, `Garage Area`)
          # it comes out they are the houses with no garage; it would be 
          # appropriate to impute NA with a new category level (e.g., "None") 
          # for these garage variables
        
    
    # Procedures to manage missing data 
        # if only 5% or less are missing in a random pattern, in a big dataset
        # almost any procedure for handling missing values yields similar results
        # if more than 80%-90% of observations of a variable are missing, 
        # often one deletes the variable
    
        # 1. to delete 
        # Create a new dataset without missing values using na.omit()  
          # ames_without_missing <- na.omit (AmesHousing::ames_raw)
        
    
        # 2. to impute missing for one var (factor) 'manually'
        library (Hmisc) 
        ames_impute <- AmesHousing::ames_raw # a copy of the data 
        sum(is.na(ames_impute$Alley)) # 2732 missing
        ames_impute$Alley <- as.factor(ames_impute$Alley) # required to be factor
        ames_impute$Alley <- impute(ames_impute$Alley) # imputes the mode
        sum(is.na(ames_impute$Alley)) # 0 mising

        # 3. to impute missing for one or all var in the dataset using the blueprint 
        # The following would build onto our ames_recipe and 
        # impute all missing values for the Gr_Liv_Area variable
        # or for all variables: 
              # with the median value:
              ames_recipe %>%
                step_medianimpute(Gr_Liv_Area) # for mode, use step_modeimpute()
              # with KNN imputation for all predictors
              ames_recipe %>%
                step_knnimpute(all_predictors(), neighbors = 6)
              # with bagged decision trees imputation
              ames_recipe %>%
                step_bagimpute(all_predictors())
        # IF if you run the lines above now, ames_recipe object changes 
        # So to make the next code running, make sure to run again the 
        # line 58-60 which is necessary to return to the original object. 
                
        # ___________
        # Latest notes: 'naniar' library aims to make it easy to summarise,
        # visualise, and manipulate missing data. You may explore this library  
        # later... 
            library(naniar)
            AmesHousing::ames_raw  %>% is.na() %>% colSums()     
            sumvar <- miss_var_summary(AmesHousing::ames_raw) 
            print(sumvar, n=50)
            sumind <- miss_case_summary(AmesHousing::ames_raw)
            print(sumind, n=50)
            # Which combinations of variables occur to be missing together?  
            gg_miss_upset(AmesHousing::ames_raw)
            # __________   
        
#__________        
# 3.4 Feature filtering 
#__________            
    library(caret)
    caret::nearZeroVar(ames_train, saveMetrics = TRUE) %>% 
      tibble::rownames_to_column() %>% 
      filter(nzv)
    
    # We can add step_zv() and step_nzv() to our ames_recipe 
    # to remove zero or near-zero variance features 
      ames_recipe %>%
        step_nzv(all_nominal(), -all_outcomes())
      #or
        step_nzv(all_numeric(), -all_outcomes())
      # or 
        step_nzv(all_predictors())
        
    # Other feature filtering methods include using advanced modelling 
    # techniques like 'random forest' (ML2) to select the most important 
    # predictors
    
#__________    
# 3.5 Numeric feature engineering
#__________
    # Display histograms
    library(DataExplorer)
    plot_histogram(ames_train)
    
    # Outlier detection - not covered here - see at the end of this script 
    # several approaches to deal with outliers 
    
    
    # Normalize all numeric columns using the blueprint
      ames_recipe %>%
        step_YeoJohnson(all_numeric())  
    # Standardize all numeric values using the blueprint
      ames_recipe %>%
        step_center(all_numeric(), -all_outcomes()) %>%
        step_scale(all_numeric(), -all_outcomes())
    
#__________          
# 3.6. Categorical feature engineering
#__________
    # display histograms
      plot_bar(ames_train)   
      
    # for some factors, some levels that have very few observations e.g.
    count(ames_train, Neighborhood) %>% arrange(n)
    count(ames_train, Screen_Porch) %>% arrange(n)
    
    # 1. Lumping (collapsing)
    # to collapse all levels that are observed in less than 10% of the
    # training sample into an “other” category, use step_other()
    # examples
    lumping <- ames_recipe %>%
      step_other(Neighborhood, threshold = 0.01, 
                 other = "other") %>%
      step_other(Screen_Porch, threshold = 0.1, 
                 other = ">0")
          # To see the effect, we can apply the blueprint to training data 
          apply_2_training <- prep(lumping, training = ames_train) %>%
          bake(ames_train)
          # see new distribution of Neighborhood
          count(apply_2_training, Neighborhood) %>% arrange(n)
          # see new distribution of Screen_Porch
          count(apply_2_training, Screen_Porch) %>% arrange(n)
 
          
    # 2. Dummy encoding (step_dummy)
    ames_recipe %>%
      step_dummy(all_nominal(), one_hot = FALSE)
    
    # 3. Label encoding (step_integer)
    # We should be careful with label encoding unordered categorical features 
    # because most models will treat them as ordered numeric features. 
    # If a categorical feature is naturally ordered then label encoding 
    # is a natural choice
    count(ames_train, Overall_Qual)
    # Label encoded for Overall_Qual using the recipe and step_integer()
    ames_recipe %>%
      step_integer(Overall_Qual) %>%
      prep(ames_train) %>%
      bake(ames_train) %>%
      count(Overall_Qual)
    
#__________    
# 3.7. Dimension reduction (step_pca) - this topic is covered in depth later
#__________    
    # In a few words, we want to reduce the dimension of our features 
    # with principal components analysis, retain the number of components 
    # required to explain, say, 95% of the variance and use these components 
    # as features in downstream modeling
    recipe(Sale_Price ~ ., data = ames_train) %>%
      step_pca(all_numeric(), threshold = .95)
    
    
#_________________________________________________________________________  
# 3.8. A full-implementation of the feature engineering tasks (all at once)
# ________________________________________________________________________
    
    # _____________
    # Alternative 2
    #______________
    # Following HOM, the order of the tasks for this data are: 
      # Remove near-zero variance features that are categorical (nominal)
      # Ordinal encode our quality-based features (which are inherently ordinal)
      # Center and scale (i.e., standardize) all numeric features.
      # Perform dimension reduction by applying PCA to all numeric features
      # Note: this order may change in other applications 
    
          #  Create blueprint
            ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
              step_nzv(all_nominal())  %>%
              step_integer(matches("Qual|Cond|QC|Qu")) %>%
              step_center(all_numeric(), -all_outcomes()) %>%
              step_scale(all_numeric(), -all_outcomes()) %>%
              step_pca(all_numeric(), -all_outcomes())
            ames_recipe
          # Prepare: estimate feature engineering parameters based on training data
            prepare <- prep(ames_recipe, training = ames_train)
            prepare
          # bake: apply the recipe to new data (e.g., the training data or future test data) with bake()
            baked_train <- bake(prepare, new_data = ames_train)
            baked_test <- bake(prepare, new_data = ames_test)
            baked_train
            baked_test
          # now we can use these two new datasets for modelling
    
      # _____________
      # Alternative 3
      #______________
      # To develop the blueprint within each resample iteration, 
      # we only need to specify the blueprint 
      # and integrated it with caret
            
        set.seed(123)
        # Create the blueprint for train data
          ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
          step_nzv(all_predictors()) %>%
          step_integer(matches("Qual|Cond|QC|Qu")) %>%
          step_center(all_numeric(), -all_outcomes()) %>%
          step_scale(all_numeric(), -all_outcomes()) %>%
          step_dummy(all_nominal(), -all_outcomes(), one_hot = FALSE)
        
        # Next, apply the same resampling method and hyperparameter search grid 
        # as done in KNN reg application.Supply the blueprint (ames_recipe) 
        # as the first argument in train() and caret library takes care of 
        # the rest.
        
        # Re-sampling plan
        cv <- trainControl(
          method = "repeatedcv", 
          number = 10, 
          repeats = 2)
        
        # Grid of hyperparameter values
        hyper_grid <- expand.grid(k = seq(1, 10, by = 1))
        
        # The model
        knn_fit2 <- train(
          ames_recipe,      # here we supply the blueprint
          data = ames_train, 
          method = "knn", 
          trControl = cv, 
          tuneGrid = hyper_grid,
          metric = "RMSE"
        )
        
        knn_fit2 # it takes a few minutes to converge!
        
        # Looking at our results we see that, for this run, the best KNN-model 
        # was associated with k=7 with a cross-validated RMSEA of 33318.22
        # Notice in the KNN_Regression.R file (without feature engineering),
        # we obtain the best model with k=7, and cross-validated RMSE of 43607.26.
        # Hence feature engineering reduced significantly the error.PS: given 
        # the small dataset, you may obtain a slightly different result
        
        # visual display of RMSEA vs. k
        ggplot(knn_fit2)
           
        
 # Apply recipe to new data (test)
        predictions <- predict(knn_fit2, ames_test) # Since the 
        # recipe is inside of model, we do not give predict() 
        # for baked data; instead we use predict() and the original test set.
        predictions 
        # the test RMSEA 
        RMSEA = sqrt(sum((ames_test$Sale_Price - predictions)^2/877))
        RMSEA
        #or simply
        library(Metrics)
        rmse(ames_test$Sale_Price, predictions)
     
        
        
# ____________________________________________________________    
 # Extra code - not covered in HOM : Outlier detection
# ____________________________________________________________
# This is done for each var at the very beginning 
# Consider the variable: ames_train$Lot_Frontage and investigate possible outliers
# Below we present several methods
        
      library(gridExtra)    
        
        # 1. histogram, Q-Q plot, and boxplot
        par(mfrow = c(1, 3))
        hist(ames_train$Lot_Frontage, main = "Histogram")
        boxplot(ames_train$Lot_Frontage, main = "Boxplot")
        qqnorm(ames_train$Lot_Frontage, main = "Normal Q-Q plot")
        
        # 2. Mean and SD
        # get mean and standard deviation
        mean = mean(ames_train$Lot_Frontage)
        std = sd(ames_train$Lot_Frontage)
        
        # get threshold values for outliers
        Tmin = mean-(3*std)
        Tmax = mean+(3*std)
        
        # find outliers
        ames_train$Lot_Frontage[which(ames_train$Lot_Frontage < Tmin | ames_train$Lot_Frontage > Tmax)]
        # [1] 313 195 200 160 174
        
        # remove outliers
        ames_train$Lot_Frontage[which(ames_train$Lot_Frontage > Tmin & ames_train$Lot_Frontage < Tmax)]
        
        # 3. Median
        # get median
        med = median(ames_train$Lot_Frontage)
        # subtract median from each value of x and get absolute deviation
        abs_dev = abs(ames_train$Lot_Frontage-med)
        # get Median Absolute Deviation (MAD)
        mad = 1.4826 * median(abs_dev) # 1.4826 is a set value when data is normally distributed
        
        # get threshold values for outliers
        Tmin = med-(3*mad) 
        Tmax = med+(3*mad) 
        
        # find outliers
        ames_train$Lot_Frontage[which(ames_train$Lot_Frontage < Tmin | ames_train$Lot_Frontage > Tmax)]
        # [1] 153 313 195 200 140 160 152 149 150 141 174 144 150 155 149
        
        # remove outliers
        ames_train$Lot_Frontage[which(ames_train$Lot_Frontage > Tmin & ames_train$Lot_Frontage < Tmax)]
        
        
        # 4. Dixon’s Q, Grubb and Chi-square tests
        # H0: The maximum or minimum value is not an outlier 
        # Ha: The maximum or minimum value is an outlier 
        
        library(outliers)
        dixon.test(ames_train$Lot_Frontage) # for maximum
        dixon.test(x, opposite = TRUE) # for minimum
        # Dixon test can only be used for small sample sizes (N<=30)
        
        grubbs.test(ames_train$Lot_Frontage) # for maximum
        # As the p value is significant (G = 7.7440, U = 0.9707, 
        # p-value = 6.37e-12), the maximum value 313 is an outlier.
        grubbs.test(ames_train$Lot_Frontage, opposite = TRUE) # for minimum
        # As the p value (> 0.05) is not significant (G = 1.77044, U = 0.99847,
        # p-value = 1), the minimum value 0 is not an outlier.
        
        chisq.out.test(ames_train$Lot_Frontage)
        # As the p value is significant, highest value 313 is an outlier
        chisq.out.test(ames_train$Lot_Frontage, opposite = TRUE)
        # As the p value is not significant, lowest value 0 is not an outlier
        
        # If we have more than one outlier in the dataset, 
        # then we have to perform multiple tests to remove outliers. 
        # We need to remove the outlier identified in each step and 
        # repeat the process.
        # For multivariate outlier detection, a common approach is to calculate 
        # the Mahalanobis Distance. We do not cover this into detail, but  
        # for more information on this topic see R Package ‘mvoutlier’.
  # ___________      
       
        