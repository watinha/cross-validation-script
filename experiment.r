library(C50)
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

folds <- createFolds(dataset_cv[,'Result'])
crossValidation(dataset_cv, dataset_test, folds, target_features, decision_tree, 'output/decision_tree_target.csv');
crossValidation(dataset_cv, dataset_test, folds, control_features, decision_tree, 'output/decision_tree_literature.csv');




