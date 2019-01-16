# Practical Machine Learning
## Course 8 in the [Data Science Specialization by John Hopkins University](https://www.coursera.org/specializations/jhu-data-science)
1. The Data Scientist's Toolbox
2. R Programming
3. Getting and Cleaning Data
4. Exploratory Data Analysis
5. Reproducible Research
6. Statistical Inference
7. Regression Models
8. [Practical Machine Learning](https://github.com/yanniey/Coursera_Practical_Machine_Learning)
9. Developing Data Products
10. Data Science Capstone

-----------------

## Week 1
* In and out of sample errors
* Cross Validation

## Week 2
(This week's videos take a lot of time to go through & writing notes on.)

* Caret package
* Principal Component Analysis (PCA)
* Regression with multiple features
* Plotting

## Week 3
* Decision trees and random forest
* Bagging
* Boosting

## Week 4
* Regular Expression
* Unsupervised Prediction
* Combining predictors
* Forecasting

--------------


## Week 2 Notes

### Preprossing data with caret
```{r}
library(caret)
library(kernlab)
data(spam)
inTrain<-createDataPartition(y=spam$type,p=0.75,list=FALSE)
training<-spam[inTrain,]
testing<-spam[-inTrain,]
hist(training$capitalAve,main="",xlab="avg. capital run length")
```

The histogram shows that the data are heavily skewed to the left. 

#### Standardizing the variables (so that they have `mean = 0` and `sd=1`)
```{r}
trainCapAve<-training$capitalAve
trainCapAveS<-(trainCapAve-mean(trainCapAve))/sd(trainCapAve)
mean(trainCapAveS)
sd(trainCapAveS)
```

#### Standardizing the test set, using mean and sd of the training set. This means that the standardized test cap will not be exactly the same as that of the training set, but they should be similar. 
```{r}
testCapAve<-testing$capitalAve
testCapAveS<-(testCapAve-mean(trainCapAve))/sd(trainCapAve)
mean(testCapAveS)
```

#### Use preprocess() function to do the standardization on the training set. The result is the same as using the above functions
```{r}
preObj<-preProcess(training[,-58],method=c("center","scale"))
trainCapAveS<-predict(preObj,training[,-58])$capitalAve
mean(trainCapAveS)
sd(trainCapAveS)
```
#### Use `preProcess()` to do the same on the testing dataset. Note that `preObj` (which was created based on the training set) is also used to predict on the testing set.

Note that `mean()` is not equal to 0 on the testing set, and `sd` is not equal to 1.

```{r}
testCapAveS<-predict(preObj,testing[,-58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)
```

#### Use `preProcess()` directly when building a model

```{r}
set.seed(1)
model<-train(type ~.,data=training,preProcess=c("center","scale"),method="glm")
model
```

#### Standardising - Box-Cox Transforms

This transforms the data into normal shape - i.e. bell shape
```{r}
preObj<-preProcess(training[,-58],method=c("BoxCox"))
trainCapAveS<-predict(preObj,training[,-58])$capitalAve
par(mfrow=c(1,2))
hist(trainCapAveS)
qqnorm(trainCapAveS)
```

#### Standardization: Imputing data where it is NA using `knnImpute`

`knnImpute` uses the average of the k-nearest neighbours to impute the data where it's not available. 

```{r}
set.seed(1)

# Make some value NAs
training$capAve<-training$capitalAve
selectNA<-rbinom(dim(training)[1],size=1,prob=0.05)==1
training$capAve[selectNA]<-NA

# Impute data when it's NA, and standardize
preObj<-preProcess(training[,-58],method="knnImpute")
capAve<-predict(preObj,training[,-58])$capAve

# Standardize true values
capAveTruth<-training$capitalAve
capAveTruth<-(capAveTruth-mean(capAveTruth))/sd(capAveTruth)
```

Look at the difference at the imputed value (`capAve`) and the true value (`capAveTruth`), using `quantile()` function.

If the values are all relatively small, then it shows that imputing data works (i.e. doesn't change the dataset too much).
```{r}
quantile(capAve-capAveTruth)
```

#### Some notes on preprocessing data

* training and testing must be processed in the same way (i.e. use the same `preObj` in `predict()` function)


#### Covariate/Predictor/Feature Creation

1. Step 1: raw data -> features (e.g. free text -> data frame)
   Google "Feature extraction for [data type]"
   Examples:
   * Text files: frequency of words, frequency of phrases, frequency of capital letters
   * Images: Edges, corners, ridges
   * Webpages: # and type of images, position of elements, colors, videos (e.g. A/B testing)
   * People: Height, weight, hair color, gender etc.
   
2. Step 2: features -> new, useful features
   * more useful for some models (e.g. regression, SVM) than others( e.g. decision trees)
   * should be done **only on the training set**
   * new features should be added to data frames

3. An example of feature creation
        ```{r}
        library(ISLR)
        library(caret)
        data(Wage)
        inTrain<-createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
        training<-Wage[inTrain,]
        testing<-Wage[-inTrain,]
        ```
  * Convert factor variables to dummy variables 
    
    The `jobclass` column is chacracters, so we can convert it to dummy variable with `dummyVars` function
    
    ```{r}
    dummies<-dummyVars(wage ~ jobclass,data=training)
    head(predict(dummies,newdata=training))
    ```
    
   * Remove features which is the same throughout the dataframe, using `nearZeroVar`
    
    If nsv (`nearZeroVar`) returns TRUE, then this feature is not important and thus can be removed. 
    
    ```{r}
    nsv<-nearZeroVar(training,saveMetrics = TRUE)
    nsv
    ```
    * Spline basis
    `df=3` says that we want a 3rd-degree polynomial on this variable `training$age`.
    First column means `age`
    Second column means `age^2`
    Third column means `age^3`
    ```{r}
    library(splines)
    bsBasis<-bs(training$age,df=3)
    bsBasis
    ```
    #### Fitting curves with splines
    ```{r}
    lm1<-lm(wage~bsBasis,data=training)
    plot(training$age,training$wage,pch=19,cex=0.5)
    points(training$age,predict(lm1,newdata=training),col="red",pch=19,cex=0.5)
    ```
    #### splines on the test set.
    Note that we are using the same `bsBasis` as is created in the training dataset
    ```{r}
    predict(bsBasis,age=testing$age)
    ```

### PCA (Principal Components Analysis), mostly useful for linear-type models

1. Find features which are correlated

`which()` returns the list of features with correlation > 0.8
```{r}
library(caret)
library(kernlab)
data(spam)
set.seed(1)
inTrain<-createDataPartition(y=spam$type,p=0.75,list=FALSE)
training<-spam[inTrain,]
testing<-spam[-inTrain,]

M<-abs(cor(training[,-58]))
diag(M)<-0
which(M>0.8,arr.ind=T)
```
  
  Take a look at the correlated features:
  
```{r}
  names(spam)[c(34,32,40)]
  plot(spam[,34],spam[,32])
```

  Apply PCA in R: `prcomp()`
```{r}
smallSpam<-spam[,c(34,32)]
prComp<-prcomp(smallSpam)
plot(prComp$x[,1],prComp$x[,2])
prComp$rotation
```
 
 #### PCA on spam data
```{r}
typeColor<-((spam$type=="spam")*1+1)
prComp<-prcomp(log10(spam[,-58]+1))
plot(prComp$x[,1],prComp$x[,2],col=typeColor,xlab="PC1",ylab="PC2")
```

  #### PCA with caret, preProcess()
```{r}
preProc<-preProcess(log10(spam[,-58]+1),method="pca",pcaComp = 2)
spamPC<-predict(preProc,log10(spam[,-58]+1))
plot(spamPC[,1],spamPC[,2],col=typeColor)
```

  #### Preprocessing with PCA to create model based on the training set
```{r,warning=FALSE}
preProc<-preProcess(log10(training[,-58]+1),method="pca",pcaComp=2)
trainPC<-predict(preProc,log10(training[,-58]+1))
modelFit <- train(x = trainPC, y = training$type,method="glm")
```

  #### Preprocessing with PCA to use on the testing set
  Note that we should use the same PCA procedure (`preProc`) when using predict()
 on the testing set
```{r}
testPC<-predict(preProc,log10(testing[,-58]+1))
confusionMatrix(testing$type,predict(modelFit,testPC))
```

  Accuracy is > 0.9!
  
  #### Alternative: preProcess with PCA during the training process (instead of doing PCA first, then do the training)
  
```{r,warning=FALSE}
modelFit <- train(x = trainPC, y = training$type,method="glm",preProcess="pca")
confusionMatrix(testing$type,predict(modelFit,testPC))
```

### Predicting with Regression

Use the fainthful eruption data in caret
```{r}
library(caret)
data(faithful)
set.seed(333)
inTrain<-createDataPartition(y=faithful$waiting,p=0.5,list=FALSE)
trainFaith<-faithful[inTrain,]
testFaith<-faithful[-inTrain,]
head(trainFaith)
```

#### Plot eruption duration vs. waiting time.
You can see that there's a roughly linear relationship between the two variables. 
```{r}
plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="waiting",ylab="eruption duration")
```

#### Fit a linear regression model
```{r}
lm1<-lm(eruptions~waiting,data=trainFaith)
summary(lm1)
```

#### Plot the model fit
```{r}
plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",xlab="waiting",ylab="eruption duration")
lines(trainFaith$waiting,lm1$fitted,lwd=3)
```

#### Predicting a new value with the linear regression model
When `waiting time = 80`
```{r}
newdata<-data.frame(waiting=80)
predict(lm1,newdata)
```

#### Plot predictions - training vs testing set
```{r}
par(mfrow=c(1,2))
# training
plot(trainFaith$waiting,trainFaith$eruptions,pch=19,col="blue",main="training",xlab="waiting",ylab="eruption duration")
lines(trainFaith$waiting,predict(lm1),lwd=3)
# testing
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue",main="testing",xlab="waiting",ylab="eruption duration")
lines(testFaith$waiting,predict(lm1,newdata=testFaith),lwd=3)

```

#### Get training & testing errors
```{r}
# RMSE on training
sqrt(sum((lm1$fitted-trainFaith$eruptions)^2))
# RMSE on testing
sqrt(sum((predict(lm1,newdata=testFaith)-testFaith$eruptions)^2))
```

#### Prediction intervals
```{r}
pred1<-predict(lm1,newdata=testFaith,interval="prediction")
ord<-order(testFaith$waiting)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue")
matlines(testFaith$waiting[ord],pred1[ord,],type="l",col=c(1,2,2),lty=c(1,1,1),lwd=3)
```

#### Same process with caret
```{r}
modFit<-train(eruptions~waiting,data=trainFaith,method="lm")
summary(modFit$finalModel)
```

## Predicting with regression, multiple covariates
Use the wages dataset in ISLR package
```{r}
library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
Wage<-subset(Wage,select=-c(logwage))
summary(Wage)

inTrain<-createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
training<-Wage[inTrain,]
testing<-Wage[-inTrain,]
dim(training)
dim(testing)
```

#### Feature plot on the wages dataset
```{r}
featurePlot(x=training[,c("age","education","jobclass")],y=training$wage,plot="pairs")
```

#### Plot age vs. wage
```{r}
qplot(age,wage,data=training)
```

#### Plot age vs wage, color by jobclass
We can see that the outliners are mostly for people in informational jobclass
```{r}
qplot(age,wage,color=jobclass,data=training)
```

#### Plot age vs. wage, color by education
You can see that the outliners are mostly advance degree education
```{r}
qplot(age,wage,color=education,data=training)
```

#### Fit a linear model
```{r}
modFit<-train(wage~age+jobclass+education,method="lm",data=training)
finMod<-modFit$finalModel
print(modFit)

plot(finMod,1,pch=19,cex=0.5,col="#00000010")
```


#### Color by variables not used in the model
```{r}
qplot(finMod$fitted,finMod$residuals,color=race,data=training)
```

#### Plot by index (i.e. which rows in the dataframe they are at)
```{r}
plot(finMod$residuals,pch=19)
```

#### Predicted vs. truth in test set
```{r}
pred<-predict(modFit,testing)
qplot(wage,pred,color=year,data=testing)
```

### If you want to use all covariates (variables)
```{r}
modFitAll<-train(wage~.,data=training,method="lm")
pred<-predict(modFitAll,newdata=testing)
qplot(wage,pred,data=testing)
```


## Week 3

## Regression with Trees
Pros: better interpretability, better performance for non-linear settings

Stop splitting when the leaves are pure 
### Measures of impurity
1. Misclassification Error:
  * 0 = perfect purity
  * 0.5 = no purity

2. Gini index:
  * 0 = perfect purity
  * 0.5 = no purity

3. Deviance/information gain:
  * 0 = perfect purity
  * 1 = no purity
  
Example: Iris Data
```{r, warning=FALSE}
data(iris)
library(ggplot2)
library(caret)
names(iris)
table(iris$Species)

inTrain<-createDataPartition(y=iris$Species,p=0.7,list=FALSE)
training<-iris[inTrain,]
testing<-iris[-inTrain,]

# plot the Iris petal widths/species
qplot(Petal.Width,Sepal.Width,col=Species, data=training)
```
Train the model
```{r}
#rpart is R's package for doing regressions
modFit<-train(Species~.,method="rpart",data=training)
print(modFit$finalModel)

# plot tree
plot(modFit$finalModel,uniform=TRUE,main="Classification Tree")
text(modFit$finalModel,use.n=TRUE,all=TRUE,cex=0.8)
```
Use the rattle package to make the trees look better
```{r}
library(rattle)
fancyRpartPlot(modFit$finalModel)
```

Predict new values
```{r}
predict(modFit,newdata=testing)
```

Notes:
Classification trees are non-linear models
* They use interaction between variables
* Tree can also be used for regression problems (i.e. continuous outcome)

## Bagging (Bootstrap aggregating)
What is bagging?
1. Resample cases and recalculate predictions
2. Average or majority vote
3. It produces similar bias, but reduces variance.
4. Bagging is more useful for non-linear functions

Example with the Ozone data from ElemStatLearn package
```{r}
library(ElemStatLearn)
data(ozone,package="ElemStatLearn")
ozone<-ozone[order(ozone$ozone),]
```

We'll predict temperature based on zone

### Bagged loess
```{r}
ll<-matrix(NA,nrow=10,ncol=155)

#we'll resample the data 10 times (loop 10 times)
for(i in 1:10){
        # each time we'll resample with replacement
        ss<-sample(1:dim(ozone)[1],replace=T)
        # ozone0 is the resampled subset. We'll also reorder the resampled subset with ozone
        ozone0<-ozone[ss,];ozone0<-ozone0[order(ozone0$ozone),]
        # we'll fit a loess line through the resampled subset. span determins how smooth this line would be 
        loess0<-loess(temperature~ozone,data=ozone0,span=0.2)
        # for each of the loess curve, we'll predict the outcome for the 155 rows in the original dataset
        ll[i,]<-predict(loess0,newdata=data.frame(ozone=1:155))
}
```

### Bagged loess
The red line is the bagged (average) line across the 10 resamples
```{r}
plot(ozone$ozone,ozone$temperature,pch=19,cex=0.5)
for(i in 1:10){lines(1:155,ll[i,],col="grey",lwd=2)}
lines(1:155,apply(ll,2,mean),col="red",lwd=2)
```

Notes: 
* Bagging is most useful for non-linear models
* Often used with trees & random forests

## Random Forests
What is random forests?
1. Bootstrap samples
2. At each split, bootstrap variables
3. Grow multiple trees and vote

**Pros:**
1. Accuracy

**Cons:**
1. Speed
2. Interpretability
3. Overfitting

Random Forest on Iris data
```{r}
data(iris)
library(ggplot2)
library(caret)
inTrain<-createDataPartition(y=iris$Species,p=0.7,list=FALSE)
training<-iris[inTrain,]
testing<-iris[-inTrain,]

# build random forest model using caret
modFit<-train(Species~.,model="rf",prox=TRUE,data=training)
```

### Getting a single tree
```{r}
library(randomForest)
getTree(modFit$finalModel,k=2)
```

### Class "centers"
```{r}
irisP<-classCenter(training[,c(3,4)],training$Species,modFit$finalModel$prox)
irisP<-as.data.frame(irisP)
irisP$Species<-rownames(irisP)
p<-qplot(Petal.Width,Petal.Length,col=Species,data=training)

# This line plots the three centers
p+geom_point(aes(x=Petal.Width,y=Petal.Length,col=Species),size=5,shape=4,data=irisP)
```

### Predicting new values
```{r}
pred<-predict(modFit,testing)
testing$preRight<-pred==testing$Species
table(pred,testing$Species)
qplot(Petal.Width,Petal.Length,col=preRight,data=testing,main="newdata Predictions")
```

## Boosting
Boosting and random forest are two of the most accurate out of the box classifiers for prediction analysis.

### What is boosting?
1. Take lots of (possibly) weak predictors
2. Weight them and add them up
3. Get a strong predictor

### Wage example for boosting
```{r}
library(ISLR)
data(Wage)
library(ggplot2)
library(caret)

Wage<-subset(Wage,select=-c(logwage))
set.seed(1)
inTrain<-createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
training<-Wage[inTrain,]
testing<-Wage[-inTrain,]
```

### Fit the boosting model
`gbm` is boosting for tree models.

```{r,warning=FALSE}
modFit<-train(wage~.,data=training,method="gbm",verbose=FALSE)
qplot(predict(modFit,testing),wage,data=testing)
```

## Model based prediction

### What is model based prediction?
1. Assume the data follow a probabilistic model
2. Use Bayes' theorem to identify optimal classifiers

### Pros
1. Take advantage of data structures
2. Computationally convenient
3. Reasonably accurate

### Cons
1. Make additional assumptions about data
2. When model is incorrect, it may reduce accuracy

**Naive Bayes** assumes that all features are independent of each other - useful for binary or categorical data, e.g. text classification

Model based prediction with Iris data
```{r}
data(iris)
library(ggplot2)
library(caret)

set.seed(2)
inTrain<-createDataPartition(y=iris$Species,p=0.7,list=FALSE)
training<-iris[inTrain,]
testing<-iris[-inTrain,]
```

### Build predictions
* `lda` = linear discriminant analysis
* `nb` = Naive Bayes

```{r,warning=FALSE}
modlda<-train(Species~.,data=training,method="lda")
modnb<-train(Species~.,data=training,method="nb")
plda<-predict(modlda,testing)
pnb<-predict(modnb,testing)
table(plda,pnb)
```

```{r}
equalPredictions =(plda==pnb)
qplot(Petal.Width,Sepal.Width,col=equalPredictions,data=testing)
```

-------------------


## Week 4


## Regularized regression
### What is regularised regression?
1. Fit a regression model
2. Penalize (or shrink) large coefficients

**Pros**
1. Help with bias/variance tradeoff
2. Help with model selection

**Cons**
1. Computationally demanding
2. Lower performance than random forests and boosting 


Prediction error = irreducible error + bias^2 + variance

**Tuning parameter lambda**
1. lambda controls the size of the coefficients, and the amount of regularization 
2. As lambda approaches 0, we obtain the least squared solution (i.e. what we get from the standard linear model)

3. As lambda approaches infinity, the coefficients go towards 0

In `caret` methods for penalised regularization models are:
+ ridge
+ lasso
+ relaxo

## Combining predictors
1. Combining classifiers generally improves accuracy but reduces interpretability.

2. How? Use majority vote
* similar classifiers: bagging, boosting, random forest
* different classifiers: model stacking, model ensembling

3. example with wage data
```{r}
library(ISLR)
data(Wage)
library(ggplot2)
library(caret)
Wage<-subset(Wage,select=-c(logwage))

# Create a building data set (which is split into training and testing data) and validation set
inBuild<-createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
validation<-Wage[-inBuild,]
buildData<-Wage[inBuild,]

inTrain<-createDataPartition(y=buildData$wage,p=0.7,list=FALSE)
training<-buildData[inTrain,]
testing<-buildData[-inTrain,]

dim(training)
```

Build two different models: linear regression + random forest
```{r}
mod1<-train(wage~.,method="glm",data=training)
mod2<-train(wage~.,method="rf",data=training,trControl=trainControl(method="cv"),number=3)
```

Plot the two different models on the same chart
```{r}
pred1<-predict(mod1,testing)
pred2<-predict(mod2,testing)
qplot(pred1,pred2,colour=wage,data= testing)
```

#### Fit a model that combines two different predictors

First, create a new dataframe that is the prediction of the original two models. Then train a new model based on the new dataframe. 
```{r}
predDF<-data.frame(pred1,pred2,wage=testing$wage)
combModFit<-train(wage~.,method="gam",data=predDF)
combPred<-predict(combModFit,predDF)
```

#### Testing erros between the two original predictors and the combined predictor
```{r}
# linear regression
sqrt(sum((pred1-testing$wage)^2))

# random forest
sqrt(sum((pred2-testing$wage)^2))

# combined (linear regression + random forest)
sqrt(sum((combPred-testing$wage)^2))
```

We can see that the combined predictor has the lowest testing error rate

#### Predict on validation set
```{r}
pred1V<-predict(mod1,validation)
pred2V<-predict(mod2,validation)
predVDF<-data.frame(pred1=pred1V,pred2=pred2V)
combPredV<-predict(combModFit,predVDF)
```

#### Error rate on validation set
```{r}
# linear regression
sqrt(sum((pred1V-validation$wage)^2))
# random forest
sqrt(sum((pred2V-validation$wage)^2))
# combined 
sqrt(sum((combPredV-validation$wage)^2))

```

### Notes on combining predictors
Typical model for binary/multiclass data
* Build an odd number of models
* Predict with each model
* Predict the class by majority vote

## Forecasting on time series & spatial data
1. example: predict the price of Google stock 
```{r}
library(quantmod)
from.dat<-as.Date("01/01/08",format="%m/%d/%y")
to.dat<-as.Date("12/31/13",format="%m/%d/%y")
getSymbols("GOOG",src="yahoo",from=from.dat,to=to.dat)
```

#### Summarize monthly opening price for Google, and store it as time series
```{r}
mGoog<-to.monthly(GOOG)
googOpen<-Op(mGoog)
ts1<-ts(googOpen,frequency = 12)
plot(ts1,xlab="Year+1",ylab="GOOG")
```

#### Decompose a time series into parts
trend, seasonal and random 
```{r}
plot(decompose(ts1),xlab="Years+1")
```

#### Build training and test sets for the prediction
```{r}
ts1Train<-window(ts1,start=1,end=5)
ts1Test<-window(ts1,start=5,end=(7-0.01))
ts1Train
```

#### Simple moving average
```{r}
plot(ts1Train)
lines(ma(ts1Train,order=3),col="red")
```

#### Exponential smoothing (weights nearby time points more than points that are far away)

We can get a range of the possible 
```{r}
library(forecast)
ets1<-ets(ts1Train,model="MMM")
fcast<-forecast(ets1)
plot(fcast)
lines(ts1Test,col="red")
```