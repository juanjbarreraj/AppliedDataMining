# ============================================================
# Final Project - Taxi Cancellation Prediction
# Juan Barrera
# MIS545.EA: Applied Data Mining
# ============================================================

# ------------------------------------------------------------
# 1. Load Required Packages
# ------------------------------------------------------------

required_packages <- c(
  "caret",
  "gains",
  "rpart",
  "rpart.plot",
  "adabag"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ------------------------------------------------------------
# 2. Start Output File
# ------------------------------------------------------------

sink("FinalProject_TaxiCancellation_Output.txt")

cat("============================================================\n")
cat("Final Project - Taxi Cancellation Prediction\n")
cat("Juan Barrera\n")
cat("MIS545.EA: Applied Data Mining\n")
cat("============================================================\n\n")


# ------------------------------------------------------------
# 3. Load Dataset
# ------------------------------------------------------------

cat("============================================================\n")
cat("1. Loading Dataset\n")
cat("============================================================\n\n")

taxi.df <- read.csv("Taxi-cancellation-case.csv")

cat("First six rows of the dataset:\n")
print(head(taxi.df))

cat("\nDataset dimensions:\n")
print(dim(taxi.df))

cat("\nColumn names:\n")
print(names(taxi.df))

cat("\nStructure of the dataset:\n")
print(str(taxi.df))


# ------------------------------------------------------------
# 4. Business Target Summary
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("2. Target Variable Summary: Car_Cancellation\n")
cat("============================================================\n\n")

cat("Cancellation frequency table:\n")
print(table(taxi.df$Car_Cancellation))

cat("\nCancellation percentage table:\n")
print(round(prop.table(table(taxi.df$Car_Cancellation)) * 100, 2))

cat("\nInterpretation:\n")
cat("The target variable is Car_Cancellation.\n")
cat("A value of 0 means the booking was not cancelled.\n")
cat("A value of 1 means the booking was cancelled.\n")
cat("The distribution shows that the dataset is highly imbalanced because most bookings were not cancelled.\n")
cat("This means accuracy alone can be misleading when evaluating the models.\n\n")


# ------------------------------------------------------------
# 5. Data Preprocessing
# ------------------------------------------------------------

cat("============================================================\n")
cat("3. Data Preprocessing\n")
cat("============================================================\n\n")

# Replace missing values with zero
taxi.df[is.na(taxi.df)] <- 0

cat("Missing values after replacement:\n")
print(sum(is.na(taxi.df)))

# Convert selected numeric ID variables into factors
taxi.df$vehicle_model_id <- as.factor(taxi.df$vehicle_model_id)
taxi.df$package_id <- as.factor(taxi.df$package_id)
taxi.df$travel_type_id <- as.factor(taxi.df$travel_type_id)

# Convert target variable to numeric first, then later factor when needed
taxi.df$Car_Cancellation <- as.numeric(as.character(taxi.df$Car_Cancellation))

cat("\nVariables converted to factors:\n")
cat("vehicle_model_id, package_id, travel_type_id\n\n")


# ------------------------------------------------------------
# 6. Date Feature Engineering
# ------------------------------------------------------------

cat("============================================================\n")
cat("4. Feature Engineering\n")
cat("============================================================\n\n")

# Convert date fields
taxi.df$from_date <- as.POSIXct(taxi.df$from_date, format = "%m/%d/%Y %H:%M")
taxi.df$to_date <- as.POSIXct(taxi.df$to_date, format = "%m/%d/%Y %H:%M")
taxi.df$booking_created <- as.POSIXct(taxi.df$booking_created, format = "%m/%d/%Y %H:%M")

# Create day of week and hour variables
taxi.df$from_date_DOW <- weekdays(taxi.df$from_date)
taxi.df$from_date_Hour <- as.numeric(format(taxi.df$from_date, "%H"))
taxi.df$booking_created_DOW <- weekdays(taxi.df$booking_created)

# Replace any missing engineered values with zero
taxi.df[is.na(taxi.df)] <- 0

cat("Created new date-based variables:\n")
cat("- from_date_DOW\n")
cat("- from_date_Hour\n")
cat("- booking_created_DOW\n\n")


# ------------------------------------------------------------
# 7. Trip Length Feature Engineering
# ------------------------------------------------------------

cat("============================================================\n")
cat("5. Trip Length Calculation\n")
cat("============================================================\n\n")

dist <- function(long1, lat1, long2, lat2) {
  rad <- pi / 180
  
  a1 <- lat1 * rad
  a2 <- long1 * rad
  b1 <- lat2 * rad
  b2 <- long2 * rad
  
  dlon <- b2 - a2
  dlat <- b1 - a1
  
  a <- (sin(dlat / 2))^2 + cos(a1) * cos(b1) * (sin(dlon / 2))^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  
  R <- 6378.145
  d <- R * c
  
  return(d)
}

taxi.df$trip_length <- dist(
  taxi.df$from_long,
  taxi.df$from_lat,
  taxi.df$to_long,
  taxi.df$to_lat
)

taxi.df[is.na(taxi.df)] <- 0

cat("Trip length summary:\n")
print(summary(taxi.df$trip_length))

cat("\nInterpretation:\n")
cat("The trip_length variable estimates the distance between pickup and drop-off locations using latitude and longitude.\n")
cat("This feature may help the models because longer or shorter trips may have different cancellation patterns.\n\n")


# ------------------------------------------------------------
# 8. Select Variables for Modeling
# ------------------------------------------------------------

cat("============================================================\n")
cat("6. Variable Selection\n")
cat("============================================================\n\n")

cat("All available variables after preprocessing:\n")
print(t(t(names(taxi.df))))

# Original project selected variables:
# 3:9 = vehicle_model_id, package_id, travel_type_id, from_area_id, to_area_id, from_city_id, to_city_id
# 12,13 = online_booking, mobile_site_booking
# 20:23 = from_date_DOW, from_date_Hour, booking_created_DOW, trip_length
# 19 = Car_Cancellation

selected.vars <- c(3:9, 12, 13, 20:23, 19)

model.df <- taxi.df[, selected.vars]

# Convert engineered character variables to factors
model.df$from_date_DOW <- as.factor(model.df$from_date_DOW)
model.df$booking_created_DOW <- as.factor(model.df$booking_created_DOW)

# Make sure target is numeric for models that need numeric target
model.df$Car_Cancellation <- as.numeric(model.df$Car_Cancellation)

cat("\nVariables selected for modeling:\n")
print(names(model.df))

cat("\nFinal modeling dataset structure:\n")
print(str(model.df))


# ------------------------------------------------------------
# 9. Split Data into Training and Validation Sets
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("7. Training and Validation Split\n")
cat("============================================================\n\n")

set.seed(1)

train.ind <- sample(1:dim(model.df)[1], 0.6 * dim(model.df)[1])

train.df <- model.df[train.ind, ]
valid.df <- model.df[-train.ind, ]

cat("Training dataset dimensions:\n")
print(dim(train.df))

cat("\nValidation dataset dimensions:\n")
print(dim(valid.df))

cat("\nTraining target distribution:\n")
print(table(train.df$Car_Cancellation))

cat("\nValidation target distribution:\n")
print(table(valid.df$Car_Cancellation))


# ------------------------------------------------------------
# 10. Evaluation Function
# ------------------------------------------------------------

evaluate_model <- function(model_name, actual, predicted_prob, threshold = 0.5) {
  
  cat("\n============================================================\n")
  cat("Model Evaluation:", model_name, "\n")
  cat("============================================================\n\n")
  
  predicted_class <- ifelse(predicted_prob > threshold, 1, 0)
  
  actual_factor <- factor(actual, levels = c(0, 1))
  predicted_factor <- factor(predicted_class, levels = c(0, 1))
  
  cm <- confusionMatrix(
    predicted_factor,
    actual_factor,
    positive = "1"
  )
  
  cat("Confusion Matrix:\n")
  print(cm)
  
  cat("\nGains Table:\n")
  gain <- gains(actual, predicted_prob, groups = 10)
  print(gain)
  
  accuracy <- cm$overall["Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  precision <- cm$byClass["Precision"]
  f1 <- cm$byClass["F1"]
  
  results <- data.frame(
    Model = model_name,
    Accuracy = round(as.numeric(accuracy), 4),
    Sensitivity = round(as.numeric(sensitivity), 4),
    Specificity = round(as.numeric(specificity), 4),
    Precision = round(as.numeric(precision), 4),
    F1 = round(as.numeric(f1), 4)
  )
  
  return(list(
    confusion_matrix = cm,
    gains = gain,
    results = results,
    predicted_class = predicted_class,
    predicted_prob = predicted_prob
  ))
}


# ------------------------------------------------------------
# 11. Model 1: K-Nearest Neighbors
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("8. Model 1: K-Nearest Neighbors\n")
cat("============================================================\n\n")

set.seed(1)

knnFit <- train(
  as.factor(Car_Cancellation) ~ .,
  data = train.df,
  method = "knn",
  tuneGrid = expand.grid(.k = c(1, 10))
)

cat("KNN model results:\n")
print(knnFit)

pred_knn <- predict(knnFit, valid.df, type = "prob")[, "1"]

knn_eval <- evaluate_model(
  model_name = "K-Nearest Neighbors",
  actual = valid.df$Car_Cancellation,
  predicted_prob = pred_knn
)


# ------------------------------------------------------------
# 12. Model 2: Decision Tree
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("9. Model 2: Decision Tree\n")
cat("============================================================\n\n")

set.seed(1)

tree_model <- rpart(
  Car_Cancellation ~ .,
  data = train.df,
  method = "class"
)

cat("Decision Tree complexity parameter table:\n")
print(tree_model$cptable)

best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]

cat("\nBest CP selected for pruning:\n")
print(best_cp)

pruned_tree <- prune(tree_model, cp = best_cp)

cat("\nPruned Decision Tree Summary:\n")
print(pruned_tree)

pred_tree <- predict(pruned_tree, valid.df, type = "prob")[, "1"]

tree_eval <- evaluate_model(
  model_name = "Decision Tree",
  actual = valid.df$Car_Cancellation,
  predicted_prob = pred_tree
)


# ------------------------------------------------------------
# 13. Save Decision Tree Plot
# ------------------------------------------------------------

png("Decision_Tree_Plot.png", width = 1200, height = 800)
prp(
  pruned_tree,
  main = "Pruned Decision Tree - Taxi Cancellation Prediction",
  type = 2,
  extra = 104,
  fallen.leaves = TRUE
)
dev.off()

cat("\nDecision tree plot saved as: Decision_Tree_Plot.png\n")


# ------------------------------------------------------------
# 14. Model 3: Boosting
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("10. Model 3: Boosting\n")
cat("============================================================\n\n")

set.seed(1)

train.boost <- train.df
train.boost$Car_Cancellation.factor <- as.factor(train.boost$Car_Cancellation)

# Remove numeric target and use factor target for boosting
boost_model <- boosting(
  Car_Cancellation.factor ~ .,
  data = train.boost[, -which(names(train.boost) == "Car_Cancellation")]
)

cat("Boosting variable importance:\n")
print(boost_model$importance)

pred_boost <- predict(boost_model, valid.df)$prob[, 2]

boost_eval <- evaluate_model(
  model_name = "Boosting",
  actual = valid.df$Car_Cancellation,
  predicted_prob = pred_boost
)


# ------------------------------------------------------------
# 15. Model 4: Logistic Regression
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("11. Model 4: Logistic Regression\n")
cat("============================================================\n\n")

set.seed(1)

logit_model <- glm(
  Car_Cancellation ~ .,
  data = train.df,
  family = binomial
)

cat("Logistic Regression Summary:\n")
print(summary(logit_model))

cat("\nAttempting logistic regression prediction...\n")

pred_logit <- tryCatch(
  {
    predict(logit_model, valid.df, type = "response")
  },
  error = function(e) {
    cat("Prediction error occurred:\n")
    cat(e$message, "\n")
    return(rep(NA, nrow(valid.df)))
  }
)

if (all(is.na(pred_logit))) {
  cat("\nLogistic Regression could not produce valid predictions.\n")
  cat("This supports the limitation that logistic regression was unstable for this dataset.\n")
  
  logit_results <- data.frame(
    Model = "Logistic Regression",
    Accuracy = NA,
    Sensitivity = NA,
    Specificity = NA,
    Precision = NA,
    F1 = NA
  )
  
} else {
  logit_eval <- evaluate_model(
    model_name = "Logistic Regression",
    actual = valid.df$Car_Cancellation,
    predicted_prob = pred_logit
  )
  
  logit_results <- logit_eval$results
}


# ------------------------------------------------------------
# 16. Final Model Comparison Table
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("12. Final Model Comparison Table\n")
cat("============================================================\n\n")

model_comparison <- rbind(
  knn_eval$results,
  tree_eval$results,
  boost_eval$results,
  logit_results
)

print(model_comparison)

write.csv(
  model_comparison,
  "Model_Comparison_Table.csv",
  row.names = FALSE
)

cat("\nModel comparison table saved as: Model_Comparison_Table.csv\n")


# ------------------------------------------------------------
# 17. Save Gains Charts
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("13. Saving Gains Charts\n")
cat("============================================================\n\n")

save_gains_chart <- function(filename, actual, predicted_prob, model_name) {
  
  gain <- gains(actual, predicted_prob, groups = 100)
  
  png(filename, width = 1000, height = 700)
  
  plot(
    c(0, gain$cume.pct.of.total * sum(actual)) ~ c(0, gain$cume.obs),
    xlab = "# Cases",
    ylab = "Cumulative Cancelled Bookings Captured",
    main = paste("Gains Chart -", model_name),
    type = "l",
    col = "grey",
    lwd = 2
  )
  
  lines(
    c(0, sum(actual)) ~ c(0, length(actual)),
    lty = 2,
    lwd = 2
  )
  
  legend(
    "bottomright",
    legend = c("Model", "Baseline"),
    lty = c(1, 2),
    lwd = c(2, 2)
  )
  
  dev.off()
}

save_gains_chart(
  "Gains_Chart_KNN.png",
  valid.df$Car_Cancellation,
  pred_knn,
  "K-Nearest Neighbors"
)

save_gains_chart(
  "Gains_Chart_Decision_Tree.png",
  valid.df$Car_Cancellation,
  pred_tree,
  "Decision Tree"
)

save_gains_chart(
  "Gains_Chart_Boosting.png",
  valid.df$Car_Cancellation,
  pred_boost,
  "Boosting"
)

if (!all(is.na(pred_logit))) {
  save_gains_chart(
    "Gains_Chart_Logistic_Regression.png",
    valid.df$Car_Cancellation,
    pred_logit,
    "Logistic Regression"
  )
}

cat("Saved gains charts:\n")
cat("- Gains_Chart_KNN.png\n")
cat("- Gains_Chart_Decision_Tree.png\n")
cat("- Gains_Chart_Boosting.png\n")
cat("- Gains_Chart_Logistic_Regression.png, if logistic regression produced valid predictions\n\n")


# ------------------------------------------------------------
# 18. Final Interpretation
# ------------------------------------------------------------

cat("\n============================================================\n")
cat("14. Final Interpretation for Report\n")
cat("============================================================\n\n")

cat("This project analyzed a taxi cancellation prediction problem using data mining methods in R.\n")
cat("The main business goal was to predict whether a booking would be cancelled so that taxi companies could improve driver allocation and reduce operational inefficiency.\n\n")

cat("The dataset was highly imbalanced, with most bookings belonging to the non-cancellation class.\n")
cat("Because of this, accuracy alone is not enough to judge model quality.\n")
cat("A model can achieve high accuracy by predicting most bookings as non-cancellations, but this may not help the business identify actual cancellation cases.\n\n")

cat("The models tested included K-Nearest Neighbors, Decision Tree, Boosting, and Logistic Regression.\n")
cat("Decision Tree is useful because it is easier to interpret.\n")
cat("Boosting is useful because it combines many weak learners and can capture more complex patterns.\n")
cat("Logistic Regression was tested as a traditional baseline, but it may struggle when the data contains factor-level issues or unstable predictors.\n\n")

cat("The final report should focus on the business meaning of the results, not only the technical output.\n")
cat("The most important implication is that taxi companies should use predictive modeling to identify risky bookings, but they must evaluate models carefully because class imbalance can make accuracy misleading.\n")


# ------------------------------------------------------------
# 19. End Output File
# ------------------------------------------------------------

sink()

cat("Analysis complete.\n")
cat("Check the following files in your project folder:\n")
cat("- FinalProject_TaxiCancellation_Output.txt\n")
cat("- Model_Comparison_Table.csv\n")
cat("- Decision_Tree_Plot.png\n")
cat("- Gains_Chart_KNN.png\n")
cat("- Gains_Chart_Decision_Tree.png\n")
cat("- Gains_Chart_Boosting.png\n")
cat("- Gains_Chart_Logistic_Regression.png, if created\n")
