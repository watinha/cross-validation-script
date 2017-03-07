library(C50)
library(e1071)
library(randomForest)
library(RWeka)
library(caret)
source('lib/cross-validation.r')

dataset_cv <- read.arff('data/real-dataset.trainning.arff')
dataset_test <- read.arff('data/real-dataset.test.arff')

target_features <- c('displayed',
                     'height',
                     'width',
                     'activatorTop',
                     'activatorLeft',
                     'distance',
                     'numberElements',
                     'elements_per_size',
                     'numberWords',
                     'textNodes',
                     'words_per_textNodes',
                     'Result')

control_features <- c('distanceTop',
                      'distanceLeft',
                      'textNodes',
                      'table',
                      'list',
                      'input',
                      'widgetName',
                      'date',
                      'img',
                      'proportionNumbers',
                      'links80percent',
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

folds <- createFolds(dataset_cv[,'Result'])
print('C5.0 data collecting - decision tree...')
crossValidation(dataset_cv, dataset_test, folds, target_features, decision_tree, 'output/decision_tree_target.csv');
crossValidation(dataset_cv, dataset_test, folds, control_features, decision_tree, 'output/decision_tree_literature.csv');

print('Random forest data collecting ...')
crossValidation(dataset_cv, dataset_test, folds, target_features, random_forest, 'output/random_forest_target.csv');
crossValidation(dataset_cv, dataset_test, folds, control_features, random_forest, 'output/random_forest_literature.csv');

print('E1071 - SVM... Regularizing nominal features')

dataset_cv[, 'displayed'] <- as.numeric(dataset_cv[,'displayed']) - 1

dataset_test[, 'displayed'] <- as.numeric(dataset_test[,'displayed']) - 1

crossValidation(dataset_cv, dataset_test, folds, target_features, s_v_m, 'output/svm_target.csv');

dataset_cv[, 'table'] <- as.numeric(dataset_cv[,'table']) - 1
dataset_cv[, 'list'] <- as.numeric(dataset_cv[,'list']) - 1
dataset_cv[, 'input'] <- as.numeric(dataset_cv[,'input']) - 1
dataset_cv[, 'widgetName'] <- as.numeric(dataset_cv[,'widgetName']) - 1
dataset_cv[, 'date'] <- as.numeric(dataset_cv[,'date']) - 1
dataset_cv[, 'img'] <- as.numeric(dataset_cv[,'img']) - 1
dataset_cv[, 'proportionNumbers'] <- as.numeric(dataset_cv[,'proportionNumbers']) - 1
dataset_cv[, 'links80percent'] <- as.numeric(dataset_cv[,'links80percent']) - 1

dataset_test[, 'table'] <- as.numeric(dataset_test[,'table']) - 1
dataset_test[, 'list']  <- as.numeric(dataset_test[,'list']) - 1
dataset_test[, 'input'] <- as.numeric(dataset_test[,'input']) - 1
dataset_test[, 'widgetName'] <- as.numeric(dataset_test[,'widgetName']) - 1
dataset_test[, 'date'] <- as.numeric(dataset_test[,'date']) - 1
dataset_test[, 'img']  <- as.numeric(dataset_test[,'img']) - 1
dataset_test[, 'proportionNumbers'] <- as.numeric(dataset_test[,'proportionNumbers']) - 1
dataset_test[, 'links80percent']    <- as.numeric(dataset_test[,'links80percent']) - 1

crossValidation(dataset_cv, dataset_test, folds, control_features, s_v_m, 'output/svm_literature.csv');
