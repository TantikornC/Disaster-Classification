# 🧑‍💼 Tantikorn Chatavaraha - Data Science Portfolio

Welcome to my data science portfolio! Here, I showcase my projects and expertise in data science. This repository includes detailed descriptions, methodologies, and results of my work while taking the data science practicum course.

# Disaster Tweet Classification

A collaborative project focused on developing a machine learning model to classify tweets related to natural disasters. This project was undertaken as part of the Data Scientist Practicum course at Chulalongkorn University. The dataset used was sourced from Kaggle and included various pre-engineered features such as topic keywords and tweet locations.

## Overview

The goal of this project was to leverage machine learning and natural language processing (NLP) techniques to accurately identify disaster-related tweets. This capability can enhance emergency response efforts by providing timely information extracted from social media.

## Project Structure

### 1. Data Collection & Preparation

**Data Source:** The dataset comprised 7,613 tweets, each labeled as either disaster or non-disaster. It included features like `text`, `keyword`, `location`, and `target`.

**Data Cleaning & Preprocessing:**  
- **Text Cleaning:** The process involved text extraction. Conversion to lowercase and tokenization were performed, followed by the removal of stop words and lemmatization.
- **Feature Engineering:** The following new features were created to enrich the dataset:
  - **cleaned_text:** This feature involved cleaning the original tweet text by removing non-text elements, converting to lowercase, and applying lemmatization for consistent text analysis.
  - **cleaned_keyword:** This feature standardized and corrected the keywords associated with each tweet to ensure uniformity in analysis.
  - **has_urls:** A binary feature indicating the presence of URLs in the tweets, as URLs were observed to be a potential indicator of non-disaster content.

*Example of Data Cleaning:*  
Before: "Our Deeds are the Reason of this #earthquake May allah forgive us all."  
After: "deeds reason earthquake may allah forgive us"

### 2. Exploratory Data Analysis (EDA)

The EDA phase involved a comprehensive analysis of the dataset to uncover patterns and insights that would inform the feature engineering and modeling stages. Key visualizations and analyses included:

**Word Cloud Visualizations:**  
- **Keywords:** A word cloud was generated from the `keyword` column to identify the most common keywords associated with disaster-related tweets. This visualization highlighted terms such as "hostage," "derailment," "flood," "forest fire," and "typhoon." The size of each word in the cloud represents its frequency, with larger words indicating more common keywords in the dataset. This visualization emphasized the variety and prevalence of different disaster types discussed in the tweets, providing a visual representation of the critical themes identified during data exploration.
  
<p align="center">
  <img src="./assets/images/wordcloud_keywords.png" alt="Word Cloud for Keywords"/>
</p>

- **Locations:** Another word cloud was created from the `location` column, showcasing the geographical areas most mentioned in disaster-related tweets. Locations like "California", "Texas", and "New York" were prominently featured, indicating areas frequently impacted by the reported events.

  Key observations from the location word cloud:
  - **Prominent Locations**: The most frequently mentioned locations include "USA," "New York," "Canada," "UK," and "Nigeria," indicating high tweet activity related to disasters in these areas.
  - **Global Coverage**: The word cloud shows a wide geographic spread, including locations from various continents such as "Australia," "India," "London," and "California."
  - **Significant Cities and Countries**: Both country names (e.g., "USA," "Canada") and city names (e.g., "New York," "London") appear frequently, highlighting the urban centers often discussed in the context of disasters.
  - **Diverse Mention**: The locations range from specific cities and states to broader regions and countries, reflecting the diverse scope of the dataset in terms of geographic mentions.

<p align="center">
  <img src="./assets/images/wordcloud_locations.png" alt="Word Cloud for Locations"/>
</p>

- **Text:** A word cloud was created from the `text` column to analyze the most frequently occurring words in disaster-related tweets. Interestingly, the most prominent term is "t.co," which appears frequently due to the inclusion of links in tweets. Other significant words include "fire," "people," "suicide," "flood," "police," and "killed," reflecting the critical themes discussed in the dataset. The presence of terms like "Hiroshima," "storm," and "crash" highlights specific disasters and incidents that were heavily discussed. Additionally, words like "via," "amp," and "new" show up frequently, possibly indicating common tweet structures and phrasing.

<p align="center">
  <img src="./assets/images/wordcloud_texts.png" alt="Word Cloud for Texts"/>
</p>

**Keyword Distribution by Target:**  
- **Top 10 Keywords**: The top 10 keywords predominantly associated with disaster-related tweets (Target == 1) include "derailment," "wreckage," "outbreak," "debris," and "oil spill." These terms highlight the most frequently discussed disaster events in the dataset, with a strong focus on incidents involving significant damage or threat.

<p align="center">
  <img src="./assets/images/keyword_distribution_by_target_top10.png" alt="Keyword Distribution - Top 10"/>
</p>

- **Least 10 Keywords**: The least mentioned keywords in disaster-related tweets (Target == 1) include "blew up," "threat," "screaming," and "electrocute." These terms, while still relevant to disaster contexts, appear far less frequently in the dataset, indicating they are less commonly associated with major disaster events compared to the top keywords.

<p align="center">
  <img src="./assets/images/keyword_distribution_by_target_least10.png" alt="Keyword Distribution - Least 10"/>
</p>

**Correlation of Links and Target:**  
- The presence of URLs in tweets (`has_urls`) was analyzed to determine its correlation with the `target` variable. A bar plot showed that tweets containing URLs were less likely to be classified as disaster-related, suggesting that many informative or news-related tweets might not directly address disaster specifics.

  ![Correlation of Links and Target](./images/links_correlation.png)

**Text Length Distribution by Target:**  
- The distribution of the number of characters in tweets was plotted against the `target` variable. This analysis revealed that disaster-related tweets tended to be longer, possibly due to the need to convey more detailed information during such events.

  ![Text Length Distribution](./images/text_length_distribution.png)

**Additional Insights from EDA:**
- **Keyword Cleaning:** Through the analysis, it was discovered that certain keywords contained non-standard characters (e.g., `%20` in "forest%20fire"). These were cleaned and standardized to improve data consistency and model performance.
- **Common Themes:** The analysis showed that disaster-related tweets often contained urgent language and direct calls to action (e.g., "evacuate", "emergency"), while non-disaster tweets included more general information and casual language.

The EDA provided crucial insights that guided the subsequent steps of feature engineering and model selection. By understanding the distribution and characteristics of the data, the team could make informed decisions on how to preprocess the data and what features to engineer, ultimately improving the model's effectiveness.


### 3. Model Development

**Model Selection:**  
- Experimented with several models, including Logistic Regression, Random Forest, and a deep learning model using LSTM (Long Short-Term Memory) networks. 

**Text Vectorization:**  
- Employed TF-IDF vectorization to convert textual data into numerical format, making it suitable for machine learning models.

**Training and Validation:**  
- The dataset was split into training and test sets. The LSTM model was trained with early stopping and model checkpointing to prevent overfitting and to save the best-performing model.

**Model Evaluation:**  
- The models were evaluated based on accuracy, precision, recall, and F1-score. The LSTM model achieved an accuracy of 86% with a false negative rate of 17%, indicating some challenges in correctly identifying all disaster-related tweets.

### 4. Results & Insights

**Key Results:**  
- The LSTM model demonstrated strong performance with significant accuracy in distinguishing between disaster and non-disaster tweets. The presence of certain keywords and the inclusion of URLs were critical in improving the model's prediction capability.

**Key Insights:**  
- Tweets containing certain keywords like "earthquake", "flood", or "fire" were strong indicators of disaster-related content. The analysis also revealed that tweets with URLs were more likely to be associated with non-disaster content.

### 5. Challenges & Limitations

**Challenges:**  
- Difficulty in interpreting ambiguous tweets where the context could vary significantly.
- High false negative rate, indicating a need for further refinement in capturing disaster-related nuances.

**Limitations:**  
- The model primarily relies on text data and may not accurately classify tweets with more complex contexts or those that use non-standard language, such as slang or emojis.

### 6. Future Work

**Enhancements:**  
- Incorporate more advanced NLP techniques like BERT to better understand the context and improve classification accuracy.
- Expand the dataset with more diverse examples and potentially real-time data feeds from Twitter's API to improve the model's robustness.
- Explore the integration of multimedia content analysis, including images and videos, to enhance the classification process.

## Figures & Visualizations

**Example Visualizations:**
- **Keyword Distribution:** ![Keyword Distribution](./images/keyword_distribution.png)
- **Word Cloud for Disaster-Related Keywords:** ![Word Cloud Keywords](./images/wordcloud_keywords.png)
- **Text Length Distribution by Target:** ![Text Length Distribution](./images/text_length_distribution.png)

## Contributors

- [Your Name]
- [Classmate 1]
- [Classmate 2]

## Acknowledgements

This project was part of the Data Scientist Practicum course at Chulalongkorn University. Special thanks to our instructors and peers for their guidance and support throughout the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
