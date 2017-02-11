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
crossValidation(dataset_cv, dataset_test, folds, target_features, decision_tree, 'output/decision_tree_target.csv');
crossValidation(dataset_cv, dataset_test, folds, control_features, decision_tree, 'output/decision_tree_literature.csv');

crossValidation(dataset_cv, dataset_test, folds, target_features, random_forest, 'output/random_forest_target.csv');
crossValidation(dataset_cv, dataset_test, folds, control_features, random_forest, 'output/random_forest_literature.csv');

crossValidation(dataset_cv, dataset_test, folds, target_features, s_v_m, 'output/svm_target.csv');
crossValidation(dataset_cv, dataset_test, folds, control_features, s_v_m, 'output/svm_literature.csv');



