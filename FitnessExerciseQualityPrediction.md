# Fitness Exercise Quality Analysis Based on Accelerometer Data
VS  
October 25, 2015  

## Overview

A prediction model is built and cross-validated based on the supplied "pml-training.csv" data file containing the outcome of the quality of exercise activity as a function of a subset of the large number of covariates based on a combination of instructions for the assignment, exploratory data analysis, and a dash of common sense.
  
The model is analyzed for its accuracy based on the cross-validation results, and then applied to the supplied "pml-testing.csv" test data.

## Methodology

### Covariate Selection

The instructions for the assignment suggest that we focus on accelerometer data.  Unfortunately, no data dictionary appears to be available that describes the input data, so we select the parameters according to the following criteria:  
1. Pick the set of variables whose names contain the string "accel".  
2. Remove the variables whose values are observed to contain a large number of empty (NA) values, as these will clearly contribute nothing, irrespective of their importance.  
3. Sanity-check the other variables that may have had a relevance even though their names did not contain the string "accel", and confirm that their values are largely missing.  
4. After running the training and cross-validation based on the selected variables, sanity-check the resulting accuracy.

### Cross-validation Methodology

The training data is split into a train and test set in an 70-30 ratio.  

Note that the "testing" data is not intended to be used for cross-validation, it is only used for final testing.

### Training Method

The "Random Forest" algorithm is used as the training method, as the problem domain is amenable to this best-practice algorithm that is observed to out-perform other algorithms in practice.  Since the algorithm is slow, the training results are cached in a file for iterative processing and reproducibility.

## Implementation And Output

Read the training and test data, subset the columns to the variables of interest, taking care to remove irrelevant variables (such as ID, user_name, etc.) that can cause interference with the Random Forest algorithm.

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pml_training = read.csv("./data/pml-training.csv");
pml_final_test = read.csv("./data/pml-testing.csv");
# do variable subsetting to produce training and final test sets
pml_acc_trainset = pml_training[, c("classe", grep("^accel_|^total_accel", names(pml_training),value=T))]
pml_acc_final_testset=pml_final_test[,grep("^accel_|^total_accel_", names(pml_training),value=T)]
#since final test set doesn't have the classe column, add one with N/A values
classe=rep(NA, nrow(pml_acc_final_testset))
pml_acc_final_testset=cbind(classe, pml_acc_final_testset)
```

Split the training data into training and cross-validation testing sets, generate the prediction model, observe the errors, cross-validate

```r
set.seed(12345)
inTrain = createDataPartition(y=pml_acc_trainset$classe, p=0.7, list=F)
training = pml_acc_trainset[inTrain, ]
testing = pml_acc_trainset[-inTrain, ]
if (file.exists("./data/modelFit.rds")) {
        modelFit = readRDS("./data/modelFit.rds")
} else {
        modelFit = train(classe ~ ., data=training, method="rf")
        saveRDS(modelFit, "./data/modelFit.rds")
}
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
#Display the fitted model with resampling errors etc.
print(modelFit)
```

```
## Random Forest 
## 
## 13737 samples
##    16 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9328957  0.9150541  0.004255489  0.005374887
##    9    0.9247642  0.9047688  0.003957542  0.004994703
##   16    0.9069647  0.8822458  0.005045262  0.006341866
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
#Cross-validate
cvPredictions = predict(modelFit, newdata=testing)
#Display the confusion matrix to observe the accuracy of the prediction accuracy
print(confusionMatrix(cvPredictions, testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1629   56   19   28    1
##          B   11 1031   31    5   24
##          C   12   44  962   50   14
##          D   21    3   13  878    9
##          E    1    5    1    3 1034
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9404         
##                  95% CI : (0.934, 0.9463)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9245         
##  Mcnemar's Test P-Value : 8.207e-14      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9731   0.9052   0.9376   0.9108   0.9556
## Specificity            0.9753   0.9850   0.9753   0.9907   0.9979
## Pos Pred Value         0.9400   0.9356   0.8891   0.9502   0.9904
## Neg Pred Value         0.9892   0.9774   0.9867   0.9827   0.9901
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2768   0.1752   0.1635   0.1492   0.1757
## Detection Prevalence   0.2945   0.1873   0.1839   0.1570   0.1774
## Balanced Accuracy      0.9742   0.9451   0.9565   0.9507   0.9768
```

The resampling errors and prediction accuracy generated from the above look reasonable.  

Now, apply the prediction model to the final testing set and save as well as display the final predictions:

```r
finalPredictions=predict(modelFit, newdata=pml_acc_final_testset)
# save the predictions in a file, and also display the data
pml_write_files = function(x) {
        n = length(x)
        for (i in 1:n) {
                filename=paste0("problem_id_",i,".txt")
                write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
        }
}
pml_write_files(finalPredictions)
print(finalPredictions)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The end.
