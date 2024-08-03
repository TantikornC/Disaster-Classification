# Disaster Tweet Classification

A collaborative project on disaster tweet classification using machine learning, conducted as part of the Data Scientist Practicum course at Chulalongkorn University.

## Overview

This project focuses on developing a machine learning model to classify tweets related to natural disasters. The dataset, obtained from Kaggle, consists of feature-engineered data including topic keywords and locations of Twitter posts. The primary objective is to accurately distinguish between disaster-related and non-disaster-related tweets, which can be crucial for timely and effective disaster response.

## Project Structure

### 1. Data Collection & Preparation
   - **Data Source**: The dataset was sourced from Kaggle and includes a variety of features such as keywords, text, and geographical information.
   - **Preprocessing**: The data underwent cleaning to remove duplicates, handle missing values, and standardize text formats. Feature engineering was performed to extract relevant information from the tweets, such as keywords and locations.

### 2. Exploratory Data Analysis (EDA)
   - Comprehensive EDA was conducted to understand the distribution of data, identify key features, and visualize the relationship between different attributes. We utilized libraries such as Pandas, Matplotlib, and Seaborn to create informative visualizations.

### 3. Modeling & Evaluation
   - **Model Selection**: We explored various machine learning models, including Logistic Regression, Support Vector Machines (SVM), and Random Forest Classifiers. 
   - **Feature Engineering**: Key features like keywords, locations, and tweet text were used to enhance model performance.
   - **Evaluation Metrics**: Models were evaluated based on accuracy, precision, recall, and F1-score to ensure robust performance, with special emphasis on minimizing false positives and negatives.

### 4. Results & Insights
   - The final model demonstrated strong performance in classifying tweets, providing valuable insights into the identification of disaster-related content. The inclusion of engineered features significantly improved the accuracy of the predictions.

### 5. Future Work
   - Future improvements could include integrating real-time data from Twitter's API for live disaster monitoring and exploring advanced NLP techniques like BERT for better text understanding.

## Contributors
- [Your Name]
- [Classmate 1]
- [Classmate 2]

## Acknowledgements
This project was part of the Data Scientist Practicum course at Chulalongkorn University. We would like to thank our instructors and peers for their guidance and support.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
