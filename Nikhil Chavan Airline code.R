# analysis of Airline ticket Pricing
# Name : Nikhil Chavan
# Email : nschavan1996@gmail.com
# College : IIT Bombay 

#Loading Data set
airdata.df <-read.csv(paste("SixAirlines.csv", sep=""))

##Attaching Data set
attach(airdata.df)

#view data
View(airdata.df)

#summary analysis of data
library(psych)
describe(airdata.df)

library(ggplot2) # Loading required package: ggplot2

ggplot(airdata.df, aes(x = AIRLINE, fill = AIRLINE)) + geom_bar() #Seggregating different flights 

ggplot(airdata.df, aes(x =INTERNATIONAL))+ geom_bar() #Seggregating international and domestic flights

ggplot(airdata.df, aes(x = PRICE_ECONOMY)) + geom_density() #Prices of Economy and Premium tickets

#BoxPlots to visualize data better
boxplot(PRICE_ECONOMY,horizontal=TRUE,xlab="Price of economy seats",main="Average Price of economy seats")
boxplot(PRICE_PREMIUM,horizontal=TRUE,xlab="Price of premium seats",main="Average Price of permium seats")
boxplot(N,horizontal=TRUE,main="Average No.of seat")
boxplot(PRICE_ECONOMY~AIRLINE,horizontal=TRUE,xlab="Price of economy
        
        seats",ytab="Airline",main="price of economy seats in each airline",las=1)
boxplot(PRICE_PREMIUM~AIRLINE,horizontal=TRUE,xlab="Price of PREMIUM
        
        seats",ytab="Airline",main="price of Premium seats in each airline")
boxplot(N~AIRLINE,horizontal=TRUE,xlab="No.of seats",ytab="Airline",main="No.of seats in
        
        each airline seats in each airline",las=1)

boxplot(SEATS_ECONOMY~AIRLINE,horizontal=TRUE,xlab="No.ofeconomy seats",ytab="Airline",main="No.of economy seats in
        
        each airline seats in each airline",las=1)

boxplot(SEATS_PREMIUM~AIRLINE,horizontal=TRUE,xlab="No.of premium seats",ytab="Airline",main="No.of premium seats in
        
        each airline seats in each airline",las=1)



#scatterplots to understand correlations between variables

library(car) # Loading required package: car

scatterplotMatrix(formula=~PRICE_ECONOMY+SEATS_ECONOMY,cex=0.6,diagonal="histogram")
scatterplotMatrix(formula=~PRICE_PREMIUM+SEATS_PREMIUM,cex=0.6,diagonal="histogram")
scatterplotMatrix(formula=~N+SEATS_PREMIUM+SEATS_ECONOMY,cex=0.6,diagonal="histogram")



#Calculating correlations between Prices of Economy and Premium in correlation to other factors

cor.test(PRICE_ECONOMY, PITCH_ECONOMY)
cor.test(PRICE_ECONOMY, WIDTH_ECONOMY)
cor.test(PRICE_PREMIUM, PITCH_PREMIUM)
cor.test(PRICE_PREMIUM, WIDTH_PREMIUM)

#Drawing corrgram

library(corrgram)
corrgram(airdata.df, main = "corrgram of Sixairplane variables", lower.panel = panel.shade, upper.panel =
           
           panel.pie, text.panel = panel.txt,order=TRUE)

# Performing tTests

air1.df <--airdata.df[- c(1)]

cov(air1.df)


t.test(N~AIRCRAFT)
t.test(SEATS_PREMIUM~AIRCRAFT)
t.test(SEATS_ECONOMY~AIRCRAFT)
t.test(PRICE_PREMIUM~AIRCRAFT)
t.test(PRICE_ECONOMY~AIRCRAFT)


#Regression models

reg1 <-lm(PRICE_ECONOMY~FLIGHT_DURATION+SEATS_ECONOMY+PITCH_ECONOMY+WIDTH_ECONOMY+QUALITY+MONTH+AIRCRAFT+AIRLINE,data=airdata.df)
summary(reg1)  
  
fitted(reg1)
reg1$coefficients
reg1$residuals




#Dividing the Data set into Test and Training Data ste

ratio = sample(1:nrow(airdata.df), size = 0.25*nrow(airdata.df))
Test = airdata.df[ratio,] #Test dataset 25% of total
Training = airdata.df[-ratio,] #Train dataset 75% of total
dim(Training)
dim(Test)

#Generating A Multi Variable Linear Regressional Model for Price of Economy Flights

linear.mod<- lm(PRICE_ECONOMY~ PITCH_ECONOMY + WIDTH_ECONOMY + FLIGHT_DURATION + QUALITY + PRICE_RELATIVE, data = Training)
summary(linear.mod)

#the t value of Pitch_economy and quality is positive indicating that these predictors are associated with 
#Price_economy. A larger t-value indicates that that it is less likely that the coefficient is not equal to zero purely by chance.
#Again, as the p-value for Flight_Duration and Price_Relative is less than 0.05 they are both statistically significant in the multiple linear regression model for Price_Economy response variable. 
#The model's, p-value: < 2.2e-16 is also lower than the statistical significance level of 0.05, this indicates that we can safely reject the null hypothesis that the value for the coefficient is zero 
#(or in other words, the predictor variable has no explanatory relationship with the response variable).
#The model has a F Statistic of 90, which is considerably high

library(rpart)
library(randomForest)
model.forest <- randomForest(PRICE_ECONOMY~ PITCH_ECONOMY + WIDTH_ECONOMY + FLIGHT_DURATION + QUALITY + PRICE_RELATIVE, data = Training, method = "anova", 
                             ntree = 300,
                             mtry = 2, 
                             replace = F,
                             nodesize = 1,
                             importance = T)

varImpPlot(model.forest)

#From the VIF plot we see that Flight Duration and Price Relative are most important factors in predicitng Price Economy.

#We test the model using Random Forest
prediction <- predict(model.forest,Test)
rmse <- sqrt(mean((log(prediction)-log(Test$PRICE_ECONOMY))^2))
round(rmse, digits = 3)

# Evaluation metric function
#A custom root mean Square Function to evaluate the performance of our model

RMSE <- function(x,y)
{
  a <- sqrt(sum((log(x)-log(y))^2)/length(y))
  return(a)
}

#Regression Tree Model 

model <- rpart(PRICE_ECONOMY~ PITCH_ECONOMY + WIDTH_ECONOMY + FLIGHT_DURATION + QUALITY + PRICE_RELATIVE, data = Training, method = "anova")
predict <- predict(model, Test)
RMSE1 <- RMSE(predict, Test$PRICE_ECONOMY)
RMSE1 <- round(RMSE1, digits = 3)
RMSE1

#For Premium Class Tickets

#Generating A Multi Variable Linear Regressional Model for Price of Premium Flights

linear.mod<- lm(PRICE_PREMIUM~ PITCH_PREMIUM + WIDTH_PREMIUM + FLIGHT_DURATION + QUALITY + PRICE_RELATIVE, data = Training)
summary(linear.mod)

#The model has an F Statistic of 48.4 which is mediumly high
#the t value of Pitch_premium, width_premium, Price_relative and quality is positive indicating that these predictors are associated with 
#Price_Premium. A larger t-value indicates that that it is less likely that the coefficient is not equal to zero purely by chance.
#Again, as the p-value for Flight_Duration  is less than 0.05 they are both statistically significant in the multiple linear regression model for Price_Economy response variable. 
#The model's, p-value: < 2.2e-16 is also lower than the statistical significance level of 0.05, this indicates that we can safely reject the null hypothesis that the value for the coefficient is zero 
#(or in other words, the predictor variable has no explanatory relationship with the response variable).

library(rpart)
library(randomForest)
model.forest <- randomForest(PRICE_PREMIUM~ PITCH_PREMIUM + WIDTH_PREMIUM + FLIGHT_DURATION + QUALITY + PRICE_RELATIVE, data = Training, method = "anova", 
                             ntree = 300,
                             mtry = 2, #mtry is sqrt(6)
                             replace = F,
                             nodesize = 1,
                             importance = T)

varImpPlot(model.forest)

#From the VIF plot we see that Flight Duration and Price Relative are most important factors in predicitng Price Economy.

# Evaluation metric function
#A custom root mean Square Function to evaluate the performance of our model
RMSE <- function(x,y)
{
  a <- sqrt(sum((log(x)-log(y))^2)/length(y))
  return(a)
}

#Implementing the Regression Tree Model 
model <- rpart(PRICE_ECONOMY~ PITCH_ECONOMY + WIDTH_ECONOMY + FLIGHT_DURATION + QUALITY + PRICE_RELATIVE, data = Training, method = "anova")
predict <- predict(model, Test)
RMSE1 <- RMSE(predict, Test$PRICE_ECONOMY)
RMSE1 <- round(RMSE1, digits = 3)
RMSE1








