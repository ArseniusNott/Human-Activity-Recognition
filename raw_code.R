set.seed(11111)
# configure multicore
library(doMC)
registerDoMC()

library(dplyr)
library(caret)
library(ggplot2)
library(reshape2)
library(pROC)
library(MLmetrics)
library(gridExtra)

setwd("~/Projects/Courses/Coursera/Practical Machine Learning/Human-Activity-Recognition/")
train.data.raw <- read.csv(file = "data/pml-training.csv")
test.data.raw <- read.csv(file = "data/pml-testing.csv")

# initial summary of all columns
summary(train.data.raw)

# DATA CLEANING

# Select columns with complete cases:
# class of each column in dataset
col.class <- sapply(train.data.raw, class)
# unique classes
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



# divide the train.data.reduced into training and testing 75/25
training.index <- createDataPartition(y = train.data.cleaned$classe, 
                                      p = 0.75, 
                                      list = FALSE)
training <- train.data.cleaned[training.index, ]
testing <- train.data.cleaned[-training.index, ]

# CORRELATION HEATMAP
# Create correlation matrix excluding factor columns
plot.correlation.heatmap <- function(data = training, title, subtitle) {
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
  subtitle = "All Numeric Features of Reduced Training Dataset"
)

# plot correlation heatmap of training dataset with resulting principal 
# components
plot.correlation.heatmap(
  data = train.pca, 
  title = "Fig 2. Correlation Heatmap", 
  subtitle = "Numeric Features of Training Dataset with Principal Components"
)

# Principal Component Analysis
preprocess.pca <- preProcess(x = subset(training, select = -classe), 
                             method = "pca",
                             thresh = 0.95)

# 25 principal components capture 95% of the total variance
train.pca <- predict(object = preprocess.pca,
                       newdata = subset(training, select = -classe))
train.pca$classe <- training$classe

test.pca <- predict(object = preprocess.pca,
                    newdata = subset(testing, select = -classe))
test.pca$classe <- testing$classe

# MODELING

# Training parameters
training.parameters <- trainControl(method = "repeatedcv", 
                                    number = 10,
                                    repeats = 3)

# training parameters (simple)
training.parameters <- trainControl(method = "cv", 
                                    number = 10)

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
  out.of.sample.error <- (1 - confusion.matrix$overall["Accuracy"])
  
  train.and.predict.object <- list("model" = model,
                                   "predictions" = predictions,
                                   "confusion.matrix" = confusion.matrix,
                                   "out.of.sample.error" = out.of.sample.error)
  
  train.and.predict.object
}

# (1a) Support Vector Machine
# Accuracy : 0.9992
svm.model <- train(classe ~ ., 
                   data = train.pca,
                   method = "svmRadial",
                   trControl= training.parameters,
                   na.action = na.omit)
# in-sample Accuracy
in.sample.pred <- predict(object = svm.model, newdata = train.pca)
in.sample.actual <- train.pca$classe
in.sample.rmse <- 
  sum(in.sample.pred == in.sample.actual) / length(in.sample.pred)

# predict
svm.predictions <- predict(object = svm.model,
                           newdata = test.pca)

# our-of-sample Accuracy
out.of.sample.pred <- svm.predictions
out.of.sample.actual <- test.pca$classe
out.of.sample.rmse <- 
  sum(out.of.sample.pred == out.of.sample.actual) / length(out.of.sample.actual)

# Create confusion matrix
svm.confusion.matrix <- confusionMatrix(data = svm.predictions, 
                                        reference = test.pca$classe)

# (2) Decision Tree
# Accuracy : 0.3925
decision.tree.model <- train(classe ~ ., 
                             data = train.pca, 
                             method = "rpart",
                             trControl = training.parameters,
                             na.action = na.omit,
                             metric = "Accuracy")

#Predictions
decision.tree.predictions <- predict(object = decision.tree.model,
                                     newdata = test.pca)
# Print confusion matrix and results
decision.tree.confusion.matrix <- 
  confusionMatrix(data = decision.tree.predictions, 
                  reference = test.pca$classe)

# (3) Decision Forest
# Accuracy : 0.9939 
decision.forest.model <- train(classe ~ ., 
                               data = train.pca, 
                               method = "rf",
                               trControl = training.parameters,
                               na.action = na.omit)
#Predictions
decision.forest.predictions <- predict(object = decision.forest.model,
                                       newdata = test.pca)
# Print confusion matrix and results
decision.forest.confusion.matrix <- 
  confusionMatrix(data = decision.forest.predictions, 
                  reference = test.pca$classe)

# (4) Naive Bayes
# Accuracy 0.6124
# model-based prediction
naive.bayes.model <- train(classe ~ ., 
                           data = train.pca, 
                           method = "nb", 
                           trControl = training.parameters,
                           na.action = na.omit)
#Predictions
naive.bayes.predictions <- predict(object = naive.bayes.model,
                                   newdata = test.pca)
# Print confusion matrix and results
naive.bayes.confusion.matrix <- 
  confusionMatrix(data = naive.bayes.predictions, 
                  reference = test.pca$classe)

# (5) Neural Networks
# Accuracy : 0.9994
neural.network.model <- train(classe ~ ., 
                              data = train.pca, 
                              method = "nnet",
                              trControl = trainControl(method = "boot", 
                                                       number = 1,
                                                       repeats = 1),
                              na.action = na.omit,
                              trace = FALSE)
#Predictions
neural.network.predictions <- predict(object = neural.network.model,
                                      newdata = test.pca)
# Print confusion matrix and results
neural.network.confusion.matrix <- 
  confusionMatrix(data = neural.network.predictions, 
                  reference = test.pca$classe)

nnet <- train.and.predict(model.method = "nnet", train.data = train.pca, 
                  test.data = test.pca, 
                  training.parameters = trainControl(method = "boot", 
                                                     number = 1))

# (6) Gradient Boosting 
# Accuracy : 0.9957 
# boosting predictions
# gbm boosting crashes the system.
# https://github.com/topepo/caret/issues/263
gradient.boosting.model <- train(classe ~ .,
                                 data = train.pca,
                                 method = "xgbTree",
                                 trControl = training.parameters)
#Predictions
gradient.boosting.predictions <- predict(object = gradient.boosting.model,
                                      newdata = test.pca)
# Print confusion matrix and results
gradient.boosting.confusion.matrix <- 
  confusionMatrix(data = gradient.boosting.predictions, 
                  reference = test.pca$classe)


# ensembling methods
# svm, decision tree, decision forest, naive bayes, neural networks, and
# gradient boosting
ensemble.data <- data.frame(svm.predictions, 
                            decision.forest.predictions,
                            neural.network.predictions,
                            gradient.boosting.predictions,
                            classe = test.pca$classe)
ensemble.model <- train(classe ~ .,
                        data = ensemble.data,
                        method = "svmPoly",
                        tuneGrid = data.frame(degree = 1,
                                              scale = 1,
                                              C = 1))
ensemble.predictions <- predict(object = ensemble.model,
                                newdata = ensemble.data)
# Print confusion matrix and results
ensemble.confusion.matrix <- 
  confusionMatrix(data = ensemble.predictions, 
                  reference = ensemble.data$classe)
ensemble.confusion.matrix

plot.roc.curve <- function(model.name, confusion.matrix) {
  # Accuracy
  accuracy <- round(confusion.matrix$overall["Accuracy"], 2)
  
  plot(x = NA, y = NA, xlim = c(0,1), ylim = c(0,1),
       ylab = 'True Positive Rate',
       xlab = 'False Positive Rate',
       bty = 'n', 
       main = paste("ROC Curves Per Activity (", model.name, ")"),
       sub = paste("Model has an Overall Accuracy of ", accuracy))
  
  by.class <- confusion.matrix$byClass[, c("Sensitivity", 
                                           "Specificity")]
  
  for(i in 1 : dim(by.class)[1]) {
    x <- c(0, 1 - by.class[i, "Specificity"], 1)
    y <- c(0, by.class[i, "Sensitivity"], 1)
    
    lines(y ~ x, col= i + 1, lwd=2)
  }
  
  lines(x=c(0,1), c(0,1))
  
  legend(x = "bottomright", 
         legend = c("Activity A", 
                    "Activity B", 
                    "Activity C", 
                    "Activity D", 
                    "Activity E"),
         fill = 1 : dim(by.class)[1] + 1)
}

# official
plot.roc.curve.2 <- function(model.name, train.and.predict.object) {
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
         title = paste("ROC Curves Per Activity Class (", model.name, ")"),
         subtitle = paste("Model has an Overall Out-of-Sample-Error of ", 
                          out.of.sample.error))
  
  p
}

# model and predict
dtree <- train.and.predict(model.method = "rpart", train.data = training,
                           test.data = testing, 
                           training.parameters = training.parameters)

# decision tree plot of roc curve
plot1 <- plot.roc.curve.2("Decision Tree", dtree)

plot1 <- plot.roc.curve.2("Decision Tree", decision.tree.confusion.matrix)
plot2 <- plot.roc.curve.2("Decision Forest", decision.forest.confusion.matrix)
plot3 <- plot.roc.curve.2("Naive Bayes", naive.bayes.confusion.matrix)
plot4 <- plot.roc.curve.2("Gradient Boosting", gradient.boosting.confusion.matrix)
plot5 <- plot.roc.curve.2("Support Vector Machine", svm.confusion.matrix)
plot6 <- plot.roc.curve("Neural Networks", neural.network.confusion.matrix)
plot7 <- plot.roc.curve.2("Ensemble Model", ensemble.confusion.matrix)

freq.plot <- ggplot() + 
  suppressWarnings(geom_histogram(aes(x = as.numeric(test.pca$classe), fill = test.pca$classe), 
                                  stat = "count")) +
  theme(legend.position = "bottom") +
  scale_fill_manual(name = "Activity Classes", values = 2:6) +
  labs(y = 'Frequency',
       x = 'Activity Classes',
       title = paste("Fig 3. Frequency Plot Per Activity Classes"))
freq.plot

# gets legend to be used in roc curves
get.legend <- function(a.gplot) {
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

custom.legend <- get.legend(freq.plot)

grid.arrange(plot6, custom.legend, ncol = 1, nrow = 2, heights = c(10, 1))
 