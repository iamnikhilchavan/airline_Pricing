# Analysis of Airline Ticket Pricing

This project explores data analysis and modeling techniques for airline ticket pricing using R. It focuses on analyzing the relationship between various features of airline tickets (e.g., price, seat configuration, flight duration) and building predictive models for ticket prices in economy and premium classes.

---

## Features

- **Data Loading & Exploration**: Loaded and explored a dataset of airline ticket pricing and seat configurations (`SixAirlines.csv`).
- **Visualization**: 
  - Used `ggplot2` for bar plots and density plots to analyze flight types, ticket prices, and seat distributions.
  - Boxplots for understanding ticket price variations across airlines and seat configurations.
  - Scatterplot matrices for visualizing relationships between variables.
- **Statistical Analysis**: 
  - Performed correlation tests and t-tests to understand relationships between variables.
  - Created correlation diagrams to visualize relationships.
- **Predictive Modeling**:
  - Built regression models to predict economy and premium ticket prices.
  - Developed random forest and regression tree models to evaluate feature importance and model performance.
- **Evaluation**:
  - Implemented Root Mean Square Error (RMSE) for model evaluation.

---

## Workflow

1. **Data Loading**: Loaded and attached the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**:
   - Visualized ticket pricing and seat configurations using plots and boxplots.
   - Conducted correlation tests and generated corrgrams for insights.
3. **Modeling**:
   - Built multiple linear regression models for economy and premium ticket pricing.
   - Created regression tree and random forest models to predict ticket prices and identify important features.
4. **Model Evaluation**:
   - Used custom RMSE functions to assess model performance.
   - Identified key predictors like flight duration and relative price for accurate predictions.

---

## Key Insights

- Flight duration and price relative are key predictors for both economy and premium ticket pricing.
- Seat width and pitch are positively correlated with ticket prices, suggesting a relationship between seat quality and price.
- Random forest models highlighted feature importance, with flight duration being a significant factor.

---

## Technologies Used

- **Programming Language**: R
- **Libraries**: `ggplot2`, `car`, `psych`, `corrgram`, `randomForest`, `rpart`

---

## About

- **Author**: Nikhil Chavan  
- **Email**: nschavan1996@gmail.com  
- **College**: IIT Bombay  

This project demonstrates the application of statistical and machine learning techniques in R for analyzing and predicting airline ticket prices.
