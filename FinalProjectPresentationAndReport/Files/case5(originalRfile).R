#setwd("C:/Users/user/Dropbox/R book/Assignments/Ch 21 Cases")

taxi.df <- read.csv("Taxi-cancellation-case.csv")
head(taxi.df)

## pre-process
# null to zero
taxi.df[is.na(taxi.df)] <- 0

# numbers to factors
taxi.df$vehicle_model_id <- as.factor(taxi.df$vehicle_model_id)
taxi.df$package_id <- as.factor(taxi.df$package_id)
taxi.df$travel_type_id <- as.factor(taxi.df$travel_type_id)

# date to day of week + time
taxi.df$from_date <- as.POSIXlt(taxi.df$from_date, 
                             format = "%m/%d/%Y %H:%M")
taxi.df$from_date_DOW <- weekdays(taxi.df$from_date)
taxi.df$from_date_Hour <- taxi.df$from_date$hour
taxi.df$to_date <- as.POSIXlt(taxi.df$from_date, 
                              format = "%m/%d/%Y %H:%M")
taxi.df$booking_created <- as.POSIXlt(taxi.df$from_date, 
                              format = "%m/%d/%Y %H:%M")
taxi.df$booking_created_DOW <- weekdays(taxi.df$booking_created)

# compute trip length from GPS data
dist <- function (long1, lat1, long2, lat2){
  rad <- pi/180
  a1 <- lat1 * rad
  a2 <- long1 * rad
  b1 <- lat2 * rad
  b2 <- long2 * rad
  dlon <- b2 - a2
  dlat <- b1 - a1
  a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  R <- 6378.145
  d <- R * c
  return(d)
}
taxi.df$trip_length <- dist(taxi.df$from_long, taxi.df$from_lat, 
                            taxi.df$to_long, taxi.df$to_lat)



## run prediction methods
library(caret)
library(gains)

t(t(names(taxi.df)))
selected.vars <- c(3:9, 12, 13, 20:23, 19)
set.seed(1)
train.ind <- sample(1:dim(taxi.df)[1], 0.6*dim(taxi.df)[1]) 
train.df <- taxi.df[train.ind, selected.vars]
valid.df <- taxi.df[-train.ind, selected.vars]

# knn
knnFit <- train(as.factor(Car_Cancellation) ~ ., 
                data = train.df, 
                method = "knn", 
                tuneGrid = expand.grid(.k=c(1, 10)))

pred <- predict(knnFit, valid.df, type = "prob")[,2]

# tree/ random forest
library(rpart)
library(rpart.plot)

library(adabag)

rt <- rpart(Car_Cancellation ~ ., data = train.df)
pfit<- prune(rt, cp=   rt$cptable[which.min(rt$cptable[,"xerror"]),"CP"])
prp(pfit)
pred <- predict(pfit, valid.df)

train.df$Car_Cancellation.factor <- as.factor(train.df$Car_Cancellation)
boost <- boosting(Car_Cancellation.factor ~ ., data = train.df[,-14])
pred <- predict(boost, valid.df)$prob[,2]

# logistic regression
train.df <- taxi.df[train.ind, selected.vars]
reg <- glm(Car_Cancellation ~ ., data = train.df, family=binomial)
summary(reg)
pred <- predict(reg, valid.df, type = "response") 
# note error: factor vehicle_model_id has new levels 36, 70
# same happens with package_id
# fix levels: 
reg$xlevels[["vehicle_model_id"]] <- union(reg$xlevels[["vehicle_model_id"]], 
                                           levels(valid.df$vehicle_model_id))
reg$xlevels[["package_id"]] <- union(reg$xlevels[["package_id"]], 
                                           levels(valid.df$package_id))
# predict again:
pred <- predict(reg, valid.df, type = "response")



## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation, 
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)), 
                as.factor(valid.df$Car_Cancellation))
