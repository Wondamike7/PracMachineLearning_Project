# Predicting Activity Type
Michael Moskowitz  
Saturday, October 16, 2015  



## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement â a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

So, the question is to determine which form of the barbell lift the user was performing based on the readings from the various accelerometers. The assignment includes both a training dataset and a separate test dataset (without the exercise type variable) for another portion of the project. Both are included in this write-up.

## Data exploration and cleaning
I first download and load the training and testing datasets. From early exploration, there were a number of different "NA" strings present in the data, so these are included in the `read.csv` function


```r
setInternet2(use=TRUE)
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
filename_train <- "pml-training.csv"
filename_test <- "pml-testing.csv"
if(!file.exists(filename_train)) {download.file(url_train, destfile=filename_train)}
if(!file.exists(filename_test)) {download.file(url_test, destfile=filename_test)}
```


```r
train_data <- read.csv("pml-training.csv", stringsAsFactors = FALSE, header=TRUE, na.strings=c("#DIV/0!","NA",""))
test_data <- read.csv("pml-testing.csv", stringsAsFactors = FALSE, header=TRUE, na.strings=c("#DIV/0!","NA",""))
```

With the full data sets read in, we look at the size and shape of the data. There are 19622 rows and 160 variables in the training set; the testing set has the same number of variables but only 20 rows. 

The figure below shows the distribution of the `classe` variable in the training set. This is the variable that provides the exercise type, with the five forms represented as A, B, C, D, or E. As you'll see, there is a relatively even distribution, though there are more 'A' types than the others. It does not appear skewed enough to disrupt the analysis.

```r
barplot(table(train_data$classe))
```

![](Write-up_1016_files/figure-html/init_expl-1.png) 

As mentioned, there were a number of missing values identified in the data. Exploration of the dataset identified that several variables were missing for the majority of observations, so I remove those variables completely from the dataset. There are also a number of columns in the dataset with no predictive value, such as the timestamp. These are also removed from both the training and testing datasets to create a tidy set for the analysis.


```r
goodCols <- colSums(is.na(train_data))<19215 
train_set <- train_data[,goodCols==TRUE]
test_set <- test_data[,goodCols==TRUE]
train_set <- train_set[,8:length(train_set)]
test_set <- test_set[,8:length(test_set)]

train_set$classe <- as.factor(train_set$classe)
```
For the training dataset, we still have 19622 rows, but now we are down to 53 columns, including the dependent variable, `classe`.

## Prediction model
Based on the large number of variables and the desire to predict but not necessarily interpret the model, I chose to use a Random Forest model, opting for the `randomForest` package (rather than `caret`) due to performance on my machine.

For the next portions of the paper, I will focus on my training dataset, which I split into its own training and testing sets, using a 70% / 30% split, respectively.


```r
set.seed(851)
inTrain <- createDataPartition(y=train_set$classe, p=0.7, list=FALSE)
training <- train_set[inTrain,] 
testing <- train_set[-inTrain,] 
```

We now test the model, using all the available data to estimate the `classe` variable.

```r
rf1 <- randomForest(classe ~ ., data= training, ntree = 500)
```

The resulting model has an error rate of only 0.55%, and therefore an accuracy of 99.45%

The confusion matrix from the model follows. 

```r
rf1$confusion
```

```
##      A    B    C    D    E class.error
## A 3901    3    0    1    1     0.00128
## B   18 2635    5    0    0     0.00865
## C    0   12 2379    5    0     0.00710
## D    0    0   20 2231    1     0.00933
## E    0    0    3    6 2516     0.00356
```

## Variable Importance
To understand the relative importance of variables, I produced a plot using the `varImpPlot` function in the `randomForest` package.

```r
varImpPlot(rf1,sort=TRUE, n.var=nrow(rf1$importance))
```

![](Write-up_1016_files/figure-html/varImp-1.png) 

Not shown here, but I separately used the `rfcv` function to determine variable importance, and it showed that the model with only 26 variables was nearly as accurate as the full model with all 52 variables. However, I stick with the full model as prediction accuracy is the key focus, and the effect on computing time of the shift is minimal in this case.

## Cross Validation
Though I presented the error rate from this single model, I also used k-fold cross validation to get a truer estimate of the out-of-sample rate. I used five folds, using the `createFolds` function in `caret`. I ran the same randomForest model on the training set, holding out each fold in turn. 


```r
cf_train <- createFolds(training$classe, k=5, list=TRUE) ## creates 5 folds

error_set <- rep(0,5)
for(i in 1:5){
  rf_k <- randomForest(classe ~ ., data=training[-cf_train[[i]],], ntree=500)
	error_set[i] <- rf_k$err.rate[nrow(rf_k$err.rate)]
}	
error_mn <- mean(error_set)

error_set
```

```
## [1] 0.00719 0.00755 0.00837 0.00692 0.00746
```
After each model was run, I saved the estimated error rate into a vector presented above. The mean of these errors gives us a better estimate of the out-of-sample error rate, and the result is 0.75%.

## Prediction
The final section of the paper will discuss using the model developed on the training set to predict the values in the testing set. First, I'll discuss my within sample testing (the 30% of the dataset kept separate), and then I'll discuss the predictions on the separate 20-observation test set.

### Testing sub-sample

```r
test_pred <- predict(rf1, testing, class="response")
pred_right <- test_pred==testing$classe
table(test_pred,testing$classe)
```

```
##          
## test_pred    A    B    C    D    E
##         A 1674    2    0    0    0
##         B    0 1134    8    0    0
##         C    0    3 1018    8    1
##         D    0    0    0  955    3
##         E    0    0    0    1 1078
```

To calculate the accuracy, I add up all the correct predictions and divide by the total number of observations. The result is an accuracy level of 99.56%. This is very close to the predicted accuracy level from the k-fold cross-validation of 99.25%.

### Separate test dataset
Finally, the 20-observation test set is used. I predict exercise type using the Random Forest model generated above, and provide a vector of the answers. These were submitted for the second portion of the project.

```r
answers <- predict(rf1,test_data[,-length(test_data)])
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
