library(C50)
library(e1071)
library(randomForest)
library(RWeka)
library(caret)
source('lib/cross-validation.r')

dataset_cv <- read.arff('data/dataset.arff')

features1 <- c('height',
               'width',
               'top',
               'left',
               'parent-top',
               'parent-left',
               'prev-sibling-top',
               'prev-sibling-left',
               'next-sibling-top',
               'next-sibling-left',
               'height1',
               'width1',
               'top1',
               'left1',
               'parent-top1',
               'parent-left1',
               'prev-sibling-top1',
               'prev-sibling-left1',
               'next-sibling-top1',
               'next-sibling-left1',
               'height2',
               'width2',
               'top2',
               'left2',
               'parent-top2',
               'parent-left2',
               'prev-sibling-top2',
               'prev-sibling-left2',
               'next-sibling-top2',
               'next-sibling-left2',
               'chi-squared',
               'chi-squared-1',
               'chi-squared-2',
               'diff',
               'diff-1',
               'diff-2',
               'Result')

decision_tree <- function (X, y, X_cv, y_cv) {
    model <- C5.0(X, y)
    cv_predictions <- predict.C5.0(model, X_cv)
    return (table(y_cv, cv_predictions))
}
random_forest <- function (X, y, X_cv, y_cv) {
    model <- randomForest(X, y)
    cv_predictions <- predict(model, X_cv)
    return (table(y_cv, cv_predictions))
}
s_v_m <- function (X, y, X_cv, y_cv) {
    model <- svm(X, y)
    cv_predictions <- predict(model, X_cv)
    return (table(y_cv, cv_predictions))
}

folds <- createFolds(dataset_cv[,'Result'], k=100)
errors <- c()

print('C5.0 data collecting - decision tree... - Learning curve')
for (i in 1:100) {
    partial_dataset <- dataset[]
    r <- crossValidation(dataset_cv, dataset_cv, folds, features1, decision_tree, 'output/decision_tree_target.csv');
}
