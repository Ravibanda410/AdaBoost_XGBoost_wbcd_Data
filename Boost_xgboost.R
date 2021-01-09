## ADABoosting
library(readr)
wbcd = read.csv(file.choose())
View(wbcd)

## EDA
## Removing the ID columns.
wbcd <- wbcd[,-1]
summary(wbcd)
str(wbcd)

colnames(wbcd)

## recode diagnosis as a factor
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))

## table wbcd data
table(wbcd$diagnosis)
## Propotional table for wbcd data
prop.table(table(wbcd$diagnosis))

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

## Apply  normalized function to wbcd dataset
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
View(wbcd_n)


## spliting the data into train and test data using the set.seed
library(caTools)
set.seed(0)
split <- sample.split(wbcd$diagnosis, SplitRatio = 0.8)
train <- subset(wbcd, split == TRUE)
View(train)
test <- subset(wbcd, split == FALSE)
View(test)

summary(diabeties_train)

## Building the model using the ADAboosting 
install.packages("adabag")
library(adabag)
model <- boosting(wbcd$diagnosis ~ . , data = wbcd)

## Evoluting on the test data
pred <- predict(model, test)
pred

## Matrics for test data
table(pred, test$diagnosis)

## Accuracy
test_acc <- mean(pred == test$diagnosis)

## Evoluting on train data
pred_t <- predict(model, train)

## Matrica for train data
table(pred_t, train$diagnosis)

## Accuracy for train data
train_acc <- mean(pred_t == train$diagnosis)

###\/\\/\/\/\/\/\/\/\/\/\/\\/\/\/\/\/\/\/\/\/\/\/\/\\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/


library(readr)
wbcd = read.csv(file.choose())

## EDA
## Removing the ID columns.
wbcd <- wbcd[,-1]
summary(wbcd)
str(wbcd)

colnames(wbcd)

## recode diagnosis as a factor
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))


## spliting the data into train and test data using the set.seed
library(caTools)
set.seed(0)
split <- sample.split(wbcd$diagnosis, SplitRatio = 0.8)
train <- subset(wbcd, split == TRUE)
View(train)
test <- subset(wbcd, split == FALSE)
View(test)

summary(train)
attach(train)

install.packages("xgboost")
library(xgboost)
str(train)

## Splitting the train data into train_y and train_x
train_y <- train$diagnosis == "1"
View(train_y)
train_x <- model.matrix(train$diagnosis ~ . -1, data = train)
View(train_x)

## Splitting the test data into test_y and test_x
test_y <- test$diagnosis == "1"
View(test_y)
test_x <- model.matrix(test$diagnosis ~ .-1, data = test)
View(test_x)

## Converting the train and test data into matrix form
# DMatrix on train
train_mt <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
test_mt <- xgb.DMatrix(data = test_x, label = test_y)


## Buielding the model by using the xgboost
# Max number of boosting iterations - nround
model <- xgboost(data = train_mt, nround = 100,
                       objective = "multi:softmax", eta = 0.7, 
                       num_class = 2, max_depth = 150)

# Prediction on test data
pred_test <- predict(model, test_mt)

## Matrics for test data
table(test_y, pred_test)

##Accuracy
mean(test_y == pred_test)

# Prediction on train data
train_pred <- predict(model, train_mt)

## Matrics for train data
table(train_y, train_pred)

## Accyraacy
mean(train_y == train_pred)

