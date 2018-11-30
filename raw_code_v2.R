# raw_code version 2

set.seed(11111)
library(dplyr)
library(caret)
library(ggplot2)
library(reshape2)
library(pROC)
library(MLmetrics)
library(gridExtra)

# set working directory
setwd("~/Projects/Courses/Coursera/Practical Machine Learning/Human-Activity-Recognition/")

# read csv
train.data.raw <- read.csv(file = "data/pml-training.csv")
test.data.raw <- read.csv(file = "data/pml-testing.csv")

### B. Clean Dataset For Training

# Data Cleaning

# Select columns with complete cases:
# class of each column in dataset
col.class <- sapply(train.data.raw, class)
# show unique classes
unique(col.class)

# check columns with an acceptable amount of values with complete cases 
# (non-null and non-empty values)
# acceptability in this case means that the number of values in the column 
# with complete cases is greater than or equal to 90% of the total observations
# in the dataset.
get.col.indexes.with.complete.cases <- function(data) {
  data %>% sapply(function(x) {
    if (is.numeric(x)) {
      # for numeric columns
      # check if a reasonable amount of values in the column have complete cases
      # if yes, return true, else return false
      if (sum(complete.cases(x)) >= (length(x) * 0.9)) {
        TRUE
      } else {
        FALSE
      }
    } else {
      # for factor columns
      # check if reasonable amount of values in the columns have complete cases
      # AND not equal to empty string
      # if yes, return true, else return false
      if (sum(complete.cases(x)) >= (length(x) * 0.9) & 
          sum(x != "") >= (length(x) * 0.9)) {
        TRUE
      } else {
        FALSE
      }
    }
  })
}
# reduced train dataset (columns with acceptably complete cases)
train.data.complete.cases.index <- 
  get.col.indexes.with.complete.cases(train.data.raw)
train.data.reduced <- train.data.raw[, train.data.complete.cases.index]


# for 20 unlabeled observations
unlabeled.data.reduced <- test.data.raw[, train.data.complete.cases.index]

# removing unnecessary covariates
train.data.reduced[1:5, 1:7]
train.data.reduced <- train.data.reduced[, -c(1:7)]
unlabeled.data.reduced <- unlabeled.data.reduced[, -c(1:7)]

# Clean Train Data
train.data.cleaned <- train.data.reduced %>%
  filter(complete.cases(.) & sum(is.na(.)) == 0)

### C. Split Data into Training and Testing
# divide the train.data.reduced into training and testing 75/25
training.index <- createDataPartition(y = train.data.cleaned$classe, 
                                      p = 0.75, 
                                      list = FALSE)
training <- train.data.cleaned[training.index, ]
testing <- train.data.cleaned[-training.index, ]

### D. Correlation Heatmap and Principal Component Analysis

# Correlation Heatmap
plot.correlation.heatmap <- function(data = training, 
                                     title, subtitle) {
  correlation <- 
    round(cor(subset(data, select = sapply(data, class) != "factor")), 2)
  # Melt data to bring the correlation values in two axis
  correlation.melted <- melt(correlation)
  ggplot(data = correlation.melted, 
         aes(x = Var1, y = Var2, fill = value, label= value)) +
    geom_raster() +
    scale_fill_gradient2(low = "steelblue", 
                         high = "darkred",
                         mid = "white") +
    theme(axis.text.x = element_text(angle=90, vjust = 0.6),
          axis.text.y = element_text(vjust=  0.6)) +
    labs(title = title, subtitle = subtitle, x = "", y = "")
}
# plot correlation heatmap of training dataset
plot.correlation.heatmap(
  data = training, 
  title = "Fig 1. Correlation Heatmap", 
  subtitle = "Numeric Features of Reduced Training Dataset"
)

preprocess.pca <- preProcess(x = subset(training, select = -classe), 
                             method = "pca",
                             thresh = 0.95)
# print the result of pca
preprocess.pca


# for training
train.pca <- predict(object = preprocess.pca,
                     newdata = subset(training, select = -classe))
train.pca$classe <- training$classe

# for testing
test.pca <- predict(object = preprocess.pca,
                    newdata = subset(testing, select = -classe))
test.pca$classe <- testing$classe

# plot correlation heatmap of training dataset with resulting principal 
# components
plot.correlation.heatmap(
  data = train.pca, 
  title = "Fig 2. Correlation Heatmap", 
  subtitle = "Numeric Features of Training Dataset with Principal Components"
)

## III. Model Building

### A. Establish Training Parameters

# training parameters use repeted cross-validation with 10 folds repeated three
# times. This is to make sure that the model generalizes well on new data and
# separate signal from noise.

training.parameters <- trainControl(method = "boot", number = 1)

### B. Model Building

# function that generates training model, prediction, confusion-matrix and 
# out-of-sample error
train.and.predict <- function(model.method = "rpart", train.data = train.pca, 
test.data = test.pca,
training.parameters = training.parameters) {
model <- list()

# train
if (model.method == "svmPoly") {
# since polynomial svm needs tuneGrid extra parameter, its model is 
# separated from others
model <- train(classe ~ ., 
data = train.data,
method = model.method,
trControl= training.parameters,
tuneGrid = data.frame(degree = 1,
scale = 1,
C = 1),
na.action = na.omit)
} else if (model.method == "nnet") {
# since neural network prints verbose messages per iteration,
# its modeling is separated from others
model <- train(classe ~ ., 
data = train.data,
method = model.method,
trControl= training.parameters,
na.action = na.omit,
trace = FALSE)
} else {
model <- train(classe ~ ., 
data = train.data,
method = model.method,
trControl= training.parameters,
na.action = na.omit)
}

# predict
predictions <- predict(object = model,
newdata = test.data)

# generate confusion matrix
confusion.matrix <- confusionMatrix(data = predictions, 
reference = test.data$classe)

# in-sample error
out.of.sample.error <- 1 - confusion.matrix$overall["Accuracy"]

train.and.predict.object <- list("model" = model,
"predictions" = predictions,
"confusion.matrix" = confusion.matrix,
"out.of.sample.error" = out.of.sample.error)

train.and.predict.object
}

# List of Model Methods to Build
# (1) Decision Tree
# (2) Decision Forest
# (3) Naive Bayes
# (4) Optimized Gradient Boosting
# (5) Polynomial Support Vector Machine
# (6) Neural Networks

# (1) Decision Tree
dtree <- train.and.predict(model.method = "rpart", train.data = train.pca,
test.data = test.pca, 
training.parameters = training.parameters)

# (2) Decision Forest
rforest <- train.and.predict(model.method = "rf", train.data = train.pca,
test.data = test.pca, 
training.parameters = training.parameters)

# (3) Naive Bayes
naivebayes <- train.and.predict(model.method = "nb", train.data = train.pca,
test.data = test.pca, 
training.parameters = training.parameters)

# (4) Optimized Gradient Boosting
xgbtree <- train.and.predict(model.method = "xgbTree", train.data = train.pca,
test.data = test.pca, 
training.parameters = training.parameters)

# (5) Polynomial Support Vector Machine
svmpoly <- train.and.predict(model.method = "svmPoly", train.data = train.pca,
test.data = test.pca, 
training.parameters = training.parameters)

# (6) Neural Networks
nnet <- train.and.predict(model.method = "nnet", train.data = train.pca,
test.data = test.pca, 
training.parameters = training.parameters)
```

## IV. Model Performance Analysis

### A. Frequency Plot

freq.plot <- ggplot() + 
suppressWarnings(geom_histogram(aes(x = as.numeric(test.pca$classe), 
fill = test.pca$classe), 
stat = "count"))+
theme(legend.position = "bottom") +
scale_fill_manual(name = "Activity Classes", values = 2:6) +
labs(y = 'Frequency',
x = 'Activity Classes',
title = paste("Fig 3. Frequency Plot Per Activity Classes"))
# print
freq.plot


### B. ROC Plot

# gets legend to be used in ROC plots
get.legend <- function(a.gplot) {
tmp <- ggplot_gtable(ggplot_build(a.gplot))
leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
legend <- tmp$grobs[[leg]]
return(legend)
}

# plot roc curves based on confusion matrices returned by the train.and.predict
# function above. 
plot.roc.curve <- function(model.name, train.and.predict.object) {
confusion.matrix <- train.and.predict.object$confusion.matrix

# Accuracy
out.of.sample.error <- round(train.and.predict.object$out.of.sample.error, 5)

by.class <- confusion.matrix$byClass[, c("Sensitivity", 
"Specificity")]

p <- ggplot(x = NA, y = NA, xlim = c(0,1), ylim = c(0,1)) + 
geom_line(aes(x = c(0, 1 - by.class[1, "Specificity"], 1), 
y = c(0, by.class[1, "Sensitivity"], 1)), 
colour = 2, lwd = 1) + 
geom_line(aes(x = c(0, 1 - by.class[2, "Specificity"], 1), 
y = c(0, by.class[2, "Sensitivity"], 1)), 
colour = 3, lwd = 1) + 
geom_line(aes(x = c(0, 1 - by.class[3, "Specificity"], 1), 
y = c(0, by.class[3, "Sensitivity"], 1)), 
colour = 4, lwd = 1) + 
geom_line(aes(x = c(0, 1 - by.class[4, "Specificity"], 1), 
y = c(0, by.class[4, "Sensitivity"], 1)), 
colour = 5, lwd = 1) + 
geom_line(aes(x = c(0, 1 - by.class[5, "Specificity"], 1), 
y = c(0, by.class[5, "Sensitivity"], 1)), 
colour = 6, lwd = 1) + 
geom_line(aes(x = c(0, 1), y = c(0, 1))) +
labs(y = 'True Positive Rate',
x = 'False Positive Rate',
title = paste("ROC Plot Per Activity Class (", model.name, ")"),
subtitle = paste("Model has an Overall Out-of-Sample-Error of ", 
out.of.sample.error))

p
}

# generate roc curves plot
dtree.plot <- plot.roc.curve("Decision Tree", dtree)
rforest.plot <- plot.roc.curve("Random Forest", rforest)
naivebayes.plot <- plot.roc.curve("Naive Bayes", naivebayes)
xgbtree.plot <- plot.roc.curve("Optimized Gradient Boosting", xgbtree)
svmpoly.plot <- plot.roc.curve("Support Vector Machine", svmpoly)
nnet.plot <- plot.roc.curve("Neural Networks", nnet)
custom.legend <- get.legend(freq.plot)
grid.arrange(dtree.plot, rforest.plot, naivebayes.plot, xgbtree.plot, 
svmpoly.plot, nnet.plot, custom.legend, ncol = 2, nrow = 4,
layout_matrix = rbind(c(1, 2), c(3, 4), c(5, 6), c(7, 7)),
heights = c(10, 10, 10, 1))

```

ROC Plots above (not really curves because they only get one middle value which is the average adjusted accuracies for each class)  represent accuracy conditioned on activity classes. The author wants to interest you to a out of sample errors for each modeling methods. As you can see, random-forest, optimized gradient boosting and support vector machines give near zero accuracy (SVM is zero because it is rounded to 5 places after the decimal point). To err on the side of accuracy, the author then used these modeling results to create an ensemble method with SVM as an aggregating algorithm.

### D. Ensemble Modeling

```{r 03_03_ensemble_modeling}
# Collate predictions from models with higher accuracy
gather.predictions <- function(model, new.data = train.pca) {
predictions <- predict(object = model, newdata = new.data)
predictions
}
# training set for ensemble models
ensemble.training <- data.frame(svm = gather.predictions(svmpoly$model), 
rforest = gather.predictions(rforest$model),
xgbtree = gather.predictions(xgbtree$model),
classe = train.pca$classe)
# testing set for ensemble models
ensemble.testing <- data.frame(svm = gather.predictions(svmpoly$model, 
new.data = test.pca), 
rforest = gather.predictions(rforest$model, 
new.data = test.pca),
xgbtree = gather.predictions(xgbtree$model, 
new.data = test.pca),
classe = test.pca$classe)

# train ensembled data using svm as aggregator
# no need to do cross validations
ensemble <- train.and.predict(model.method = "svmPoly", 
train.data = ensemble.training, 
test.data = ensemble.testing,
training.parameters = trainControl(
method = "boot", number = 1))

# print roc plot for ensemble model
grid.arrange(plot.roc.curve("Ensemble Model", ensemble), custom.legend, 
nrow = 2, heights = c(10, 1))

### E. Unsupervised Prediction Using Ensemble Model

unlabeled.data.pca <- predict(object = preprocess.pca,
                              newdata = subset(test.data.reduced, select = -problem_id))
predict(object = svmpoly, newdata = unlabeled.data.pca)

# Using ensemble model
ensemble.train <- data.frame(svm = gather.predictions(svmpoly$model)$predictions,
                             rforest = gather.predictions(rforest$model)$predictions,
                             classe = train.pca$classe)

## V. Conclusion

Ensemble models with esoteric aggregating algorithms will give us highly accurate models at the expense of interpretability. In the Human Activity Recognition, the author err on the side of accuracy concluding that it is possible to predict classes of movement based on the given data.