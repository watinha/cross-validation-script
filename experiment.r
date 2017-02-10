library(C50)
library(RWeka)
library(caret)

#dataset_cv <- read.arff('data/real-dataset.trainning.arff')
dataset_cv <- read.arff('data/t.arff')
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
metrics <- function (confusion_table) {
    tp <- confusion_table[2,2]
    tn <- confusion_table[1,1]
    fp <- confusion_table[1,2]
    fn <- confusion_table[2,1]
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    Fscore <- 2 * (precision * recall / (precision + recall))
    return (list(confusion_matrix=confusion_table, precision=precision, recall=recall, Fscore=Fscore))
}

# TRAINNING RESULTS
# Literature
X <- dataset_cv[,control_features[1:(length(control_features) - 1)]]
y <- dataset_cv[,'Result']
train_results_literature <- decision_tree(X, y, X, y)
# Target approach
X <- dataset_cv[,target_features[1:(length(target_features) - 1)]]
y <- dataset_cv[,'Result']
train_results_target <- decision_tree(X, y, X, y)

# CROSS-VALIDATION
folds <- createFolds(dataset_cv[,'Result'])
names <- names(folds)
results_literature <- c()
results_target <- c()
for (i in 1:length(names)) {
    train_fold <- dataset_cv[-folds[[names[i]]],]
    cv_fold <- dataset_cv[folds[[names[i]]],]
    # Literature
    X <- train_fold[,control_features[1:(length(control_features) - 1)]]
    y <- train_fold[,'Result']
    X_cv <- cv_fold[,control_features[1:(length(control_features) - 1)]]
    y_cv <- cv_fold[,'Result']
    results_literature[i] <- decision_tree(X, y, X_cv, y_cv)
    # Target approach
    X <- train_fold[,target_features[1:(length(target_features) - 1)]]
    y <- train_fold[,'Result']
    X_cv <- cv_fold[,target_features[1:(length(target_features) - 1)]]
    y_cv <- cv_fold[,'Result']
    results_target[i] <- decision_tree(X, y, X_cv, y_cv)
}

# SAVING RESULTS
csv <- matrix(nrow = 13, # 1 column header + 1 trainning result + 10 fold cross-validation result
              ncol = 15) # 1 column for title + 7 metrics * 2 approaches

csv[1, 2] <- 'TP'
csv[1, 3] <- 'TN'
csv[1, 4] <- 'FP'
csv[1, 5] <- 'FN'
csv[1, 6] <- 'Precision'
csv[1, 7] <- 'Recall'
csv[1, 8] <- 'F-Score'
csv[1, 9] <- 'TP'
csv[1, 10] <- 'TN'
csv[1, 11] <- 'FP'
csv[1, 12] <- 'FN'
csv[1, 13] <- 'Precision'
csv[1, 14] <- 'Recall'
csv[1, 15] <- 'F-Score'
csv[2, 1] <- 'Trainning metrics'
r <- metrics(train_results_literature)
csv[2, 2] <- r[['confusion_matrix']][2, 2]
csv[2, 3] <- r[['confusion_matrix']][1, 1]
csv[2, 4] <- r[['confusion_matrix']][1, 2]
csv[2, 5] <- r[['confusion_matrix']][2, 1]
csv[2, 6] <- r[['precision']]
csv[2, 7] <- r[['recall']]
csv[2, 8] <- r[['Fscore']]
r <- metrics(train_results_target)
csv[2, 9] <- r[['confusion_matrix']][2, 2]
csv[2, 10] <- r[['confusion_matrix']][1, 1]
csv[2, 11] <- r[['confusion_matrix']][1, 2]
csv[2, 12] <- r[['confusion_matrix']][2, 1]
csv[2, 13] <- r[['precision']]
csv[2, 14] <- r[['recall']]
csv[2, 15] <- r[['Fscore']]

for (i in 1:10) {
    r <- metrics(results_literature[i])
    csv[2 + i, 2] <- r[['confusion_matrix']][2, 2]
    csv[2 + i, 3] <- r[['confusion_matrix']][1, 1]
    csv[2 + i, 4] <- r[['confusion_matrix']][1, 2]
    csv[2 + i, 5] <- r[['confusion_matrix']][2, 1]
    csv[2 + i, 6] <- r[['precision']]
    csv[2 + i, 7] <- r[['recall']]
    csv[2 + i, 8] <- r[['Fscore']]
    r <- metrics(results_target[i])
    csv[2 + i, 9]  <- r[['confusion_matrix']][2, 2]
    csv[2 + i, 10] <- r[['confusion_matrix']][1, 1]
    csv[2 + i, 11] <- r[['confusion_matrix']][1, 2]
    csv[2 + i, 12] <- r[['confusion_matrix']][2, 1]
    csv[2 + i, 13] <- r[['precision']]
    csv[2 + i, 14] <- r[['recall']]
    csv[2 + i, 15] <- r[['Fscore']]
}

write.table(csv, file="results.csv", sep=",", qmethod='double')






