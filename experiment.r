library(C50)
library(e1071)
library(randomForest)
library(RWeka)
library(caret)
source('lib/learning-curve.r')
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
r_c5_target <- crossValidation(dataset_cv, dataset_test, folds, target_features, decision_tree, 'output/decision_tree_target.csv');
r_c5_lit <- crossValidation(dataset_cv, dataset_test, folds, control_features, decision_tree, 'output/decision_tree_literature.csv');

print('Random forest data collecting ...')
r_rf_target <- crossValidation(dataset_cv, dataset_test, folds, target_features, random_forest, 'output/random_forest_target.csv');
r_rf_lit <- crossValidation(dataset_cv, dataset_test, folds, control_features, random_forest, 'output/random_forest_literature.csv');

print('E1071 - SVM... Regularizing nominal features')

dataset_cv[, 'displayed'] <- as.numeric(dataset_cv[,'displayed']) - 1

dataset_test[, 'displayed'] <- as.numeric(dataset_test[,'displayed']) - 1

r_e1071_target <- crossValidation(dataset_cv, dataset_test, folds, target_features, s_v_m, 'output/svm_target.csv');

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

r_e1071_lit <- crossValidation(dataset_cv, dataset_test, folds, control_features, s_v_m, 'output/svm_literature.csv');

# GENERAL COMPARISON
r_precision <- matrix(nrow=14, ncol=7)
r_precision[1, 2] <- 'C50L'
r_precision[1, 3] <- 'C50T'
r_precision[1, 4] <- 'ForestL'
r_precision[1, 5] <- 'ForestT'
r_precision[1, 6] <- 'e1071L'
r_precision[1, 7] <- 'e1071T'
r_precision[2, 1] <- 'Trainning'
r_precision[3, 1] <- 'Fold 1'
r_precision[4, 1] <- 'Fold 2'
r_precision[5, 1] <- 'Fold 3'
r_precision[6, 1] <- 'Fold 4'
r_precision[7, 1] <- 'Fold 5'
r_precision[8, 1] <- 'Fold 6'
r_precision[9, 1] <- 'Fold 7'
r_precision[10, 1] <- 'Fold 8'
r_precision[11, 1] <- 'Fold 9'
r_precision[12, 1] <- 'Fold 10'
r_precision[13, 1] <- 'CV all'
r_precision[14, 1] <- 'Test'
r_precision[2:length(r_precision[,1]),2] <- r_c5_lit[2:length(r_c5_lit[,1]),6]
r_precision[2:length(r_precision[,1]),3] <- r_c5_target[2:length(r_c5_lit[,1]),6]
r_precision[2:length(r_precision[,1]),4] <- r_rf_lit[2:length(r_c5_lit[,1]),6]
r_precision[2:length(r_precision[,1]),5] <- r_rf_target[2:length(r_c5_lit[,1]),6]
r_precision[2:length(r_precision[,1]),6] <- r_e1071_lit[2:length(r_c5_lit[,1]),6]
r_precision[2:length(r_precision[,1]),7] <- r_e1071_target[2:length(r_c5_lit[,1]),6]
write.table(r_precision, 'output/precision.csv', quote=FALSE, sep=',', qmethod='double', row.names=FALSE, col.names=FALSE)
r_recall <- matrix(nrow=14, ncol=7)
r_recall[1, 2] <- 'C50L'
r_recall[1, 3] <- 'C50T'
r_recall[1, 4] <- 'ForestL'
r_recall[1, 5] <- 'ForestT'
r_recall[1, 6] <- 'e1071L'
r_recall[1, 7] <- 'e1071T'
r_recall[2, 1] <- 'Trainning'
r_recall[3, 1] <- 'Fold 1'
r_recall[4, 1] <- 'Fold 2'
r_recall[5, 1] <- 'Fold 3'
r_recall[6, 1] <- 'Fold 4'
r_recall[7, 1] <- 'Fold 5'
r_recall[8, 1] <- 'Fold 6'
r_recall[9, 1] <- 'Fold 7'
r_recall[10, 1] <- 'Fold 8'
r_recall[11, 1] <- 'Fold 9'
r_recall[12, 1] <- 'Fold 10'
r_recall[13, 1] <- 'CV all'
r_recall[14, 1] <- 'Test'
r_recall[2:length(r_recall[,1]),2] <- r_c5_lit[2:length(r_c5_lit[,1]),7]
r_recall[2:length(r_recall[,1]),3] <- r_c5_target[2:length(r_c5_lit[,1]),7]
r_recall[2:length(r_recall[,1]),4] <- r_rf_lit[2:length(r_c5_lit[,1]),7]
r_recall[2:length(r_recall[,1]),5] <- r_rf_target[2:length(r_c5_lit[,1]),7]
r_recall[2:length(r_recall[,1]),6] <- r_e1071_lit[2:length(r_c5_lit[,1]),7]
r_recall[2:length(r_recall[,1]),7] <- r_e1071_target[2:length(r_c5_lit[,1]),7]
write.table(r_recall, 'output/recall.csv', quote=FALSE, sep=',', qmethod='double', row.names=FALSE, col.names=FALSE)
r_fscore <- matrix(nrow=14, ncol=7)
r_fscore[1, 2] <- 'C50L'
r_fscore[1, 3] <- 'C50T'
r_fscore[1, 4] <- 'ForestL'
r_fscore[1, 5] <- 'ForestT'
r_fscore[1, 6] <- 'e1071L'
r_fscore[1, 7] <- 'e1071T'
r_fscore[2, 1] <- 'Trainning'
r_fscore[3, 1] <- 'Fold 1'
r_fscore[4, 1] <- 'Fold 2'
r_fscore[5, 1] <- 'Fold 3'
r_fscore[6, 1] <- 'Fold 4'
r_fscore[7, 1] <- 'Fold 5'
r_fscore[8, 1] <- 'Fold 6'
r_fscore[9, 1] <- 'Fold 7'
r_fscore[10, 1] <- 'Fold 8'
r_fscore[11, 1] <- 'Fold 9'
r_fscore[12, 1] <- 'Fold 10'
r_fscore[13, 1] <- 'CV all'
r_fscore[14, 1] <- 'Test'
r_fscore[2:length(r_fscore[,1]),2] <- r_c5_lit[2:length(r_c5_lit[,1]),8]
r_fscore[2:length(r_fscore[,1]),3] <- r_c5_target[2:length(r_c5_lit[,1]),8]
r_fscore[2:length(r_fscore[,1]),4] <- r_rf_lit[2:length(r_c5_lit[,1]),8]
r_fscore[2:length(r_fscore[,1]),5] <- r_rf_target[2:length(r_c5_lit[,1]),8]
r_fscore[2:length(r_fscore[,1]),6] <- r_e1071_lit[2:length(r_c5_lit[,1]),8]
r_fscore[2:length(r_fscore[,1]),7] <- r_e1071_target[2:length(r_c5_lit[,1]),8]
write.table(r_fscore, 'output/fscore.csv', quote=FALSE, sep=',', qmethod='double', row.names=FALSE, col.names=FALSE)

# generate Confidence Intervals
fscore_CI <- matrix(nrow=61, ncol=2)
fscore_CI[1, 1] <- 'F.Score'
fscore_CI[1, 2] <- 'Algorithm'
fscore_CI[2:11, 1] <- r_c5_lit[3:12,8]
fscore_CI[2:11, 2] <- 'C50 Literature'
fscore_CI[12:21, 1] <- r_c5_target[3:12,8]
fscore_CI[12:21, 2] <- 'C50 Target'
fscore_CI[22:31, 1] <- r_rf_lit[3:12,8]
fscore_CI[22:31, 2] <- 'Forest Literature'
fscore_CI[32:41, 1] <- r_rf_target[3:12,8]
fscore_CI[32:41, 2] <- 'Forest Target'
fscore_CI[42:51, 1] <- r_e1071_lit[3:12,8]
fscore_CI[42:51, 2] <- 'e1071 Literature'
fscore_CI[52:61, 1] <- r_e1071_target[3:12,8]
fscore_CI[52:61, 2] <- 'e1071 Target'
write.table(fscore_CI, 'output/fscore-CI.csv', sep=';', qmethod='double', row.names=FALSE, col.names=FALSE)

fscore_CI <- read.table('output/fscore-CI.csv', header=TRUE, sep=';')

anova <- aov(F.Score ~ Algorithm, fscore_CI)
summary(anova)
tuk <- TukeyHSD(anova, ordered=T)
tuk

png(file="images/confidence-intervals.png", height=900, width=800)
par(mar=c(5.1, 15, 5.1, 2))
plot(tuk, las=1)
title(xlab="Difference in F-Score means according to the machine learning approach used", line=2)

print(tuk, digits=20)
