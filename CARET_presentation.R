
library(RANN)
library(caret)
library(e1071)
library(ipred)


setwd("C:/Users/XXXX")
#load the files
df_raw <- read.csv("train.csv",header=T,sep=",", na.strings=c(" ","",NA,"[null]"))



ml_matrix_caret <- df_raw # copy base DF


## Start: To check number of missing values in each column

missing <- function(col){
  return (sum(is.na(col)))
}

sapply(ml_matrix_caret,missing)

### End 


### Do we to impute data ?? (ppt)

# Set up factors.
ml_matrix_caret$Survived <- as.factor(ml_matrix_caret$Survived)
ml_matrix_caret$Pclass <- as.factor(ml_matrix_caret$Pclass)
ml_matrix_caret$Sex <- as.factor(ml_matrix_caret$Sex)
ml_matrix_caret$Embarked <- as.factor(ml_matrix_caret$Embarked)


## Start: WITHOUT Cross Validation, using train and test data set

set.seed(123) # shuffle the data 
caret_df_idx <- createDataPartition(ml_matrix_caret$Survived, p= .6, list=FALSE) # Creation of IDs vector of subdivision

train_df_caret <- ml_matrix_caret[caret_df_idx,] # Creation of training and test df
test_df_caret <- ml_matrix_caret[-caret_df_idx,]

dim(train_df_caret)
dim(test_df_caret)

### End WITHOUT CV

## Start: To do Cross Validation into the model , Its optional, but it shoud increase the accuracy of our prediction with different test DF

train_control <- trainControl(method = "cv",
                              number = 20,
                              savePredictions = TRUE)

### End Cross Validation

model_caret <- train(Survived ~ .,
                     method = "glm", # GLM does not require other paramether for model calculation, some other methods does.
                     data = train_df_caret, # Value should be 'ml_matrix_caret'in case not do Cross Validation
                     trControl = train_control,  # No needed in case no use CV above
                     na.action = na.pass,
                     preProcess = c("knnImpute"))  ## Model creation managing NAs and other disruption to the model IN CASE
## Using GLM method , Generalized Linear Model, Type: Regression, Classification. 

summary(model_caret)


test_df_caret$predicted <- predict(model_caret,
                                   newdata=test_df_caret,
                                   na.action = na.pass)

View(test_df_caret, title = 'test_predicted')

## Start: Accuracy check

sum(test_df_caret$predicted == test_df_caret$Survived) / nrow(test_df_caret) # For train/test approach

sum(model_caret$pred$pred == model_caret$pred$obs) / nrow(model_caret$pred) 

str(model_caret$pred) # To have a look to structure of model prediction 

### End Accuracy

## Start: Creation of confusion matrix and F-Score methods

confusion_matrix <- confusionMatrix(model_caret$pred$pred,
                                    model_caret$pred$obs,
                                    positive = '1')

confusion_matrix
confusion_matrix$overall
confusion_matrix$table
confusion_matrix$byClass[['F1']] #Our F-Score
confusion_matrix$byClass

### End 
