# 🧑‍💼 Tantikorn Chatavaraha - Data Science Portfolio

Welcome to my data science portfolio! Here, I showcase my projects and expertise in data science. This repository includes detailed descriptions, methodologies, and results of my work while taking the data science practicum course.

# Disaster Tweet Classification

A collaborative project focused on developing a machine learning model to classify tweets related to natural disasters. This project was undertaken as part of the Data Scientist Practicum course at Chulalongkorn University. The dataset for this project was sourced from the [Kaggle "Natural Language Processing with Disaster Tweets" competition](https://www.kaggle.com/competitions/nlp-getting-started) and included various pre-engineered features such as topic keywords and tweet locations.

## Overview

The goal of this project was to leverage machine learning and natural language processing (NLP) techniques to accurately identify disaster-related tweets. This capability can enhance emergency response efforts by providing timely information extracted from social media.

## Project Structure

### 1. Data Collection & Preparation

**Data Source:** The dataset comprised 7,613 tweets, each labeled as either disaster or non-disaster. It included features like `text`, `keyword`, `location`, and `target`.

**Data Cleaning & Preprocessing:**  
- **Text Cleaning:** The text data underwent a cleaning process that involved extracting the relevant text content, converting it to lowercase, and performing tokenization. Stop words were removed, and lemmatization was applied to standardize the text for analysis.
- **Feature Engineering:** Several new features were engineered to enhance the dataset:
  - **cleaned_text:** This feature involved cleaning the original tweet text by removing non-text elements, converting to lowercase, and applying lemmatization for consistent text analysis.
  - **cleaned_keyword:** This feature involved standardizing and correcting the keywords associated with each tweet. Similar keywords were grouped together, and missing keywords were predicted using a Recurrent Neural Network (RNN) trained on the existing data, ensuring uniformity across the dataset.
  - **has_urls:** A binary feature indicating the presence of URLs in the tweets, as URLs were observed to be a potential indicator of disaster content.

*Example of Data Cleaning:*  
**Before:** "Our Deeds are the Reason of this #earthquake May allah forgive us all."  
**After:** "deeds reason earthquake may allah forgive us"

### 2. Exploratory Data Analysis (EDA)

The EDA phase involved a comprehensive analysis of the dataset to uncover patterns and insights that would inform the feature engineering and modeling stages. Key visualizations and analyses included:

**Word Cloud Visualizations:**  

<p align="center">
  <img src="./assets/images/wordcloud_keywords.png" alt="Word Cloud for Keywords"/>
</p>

<p align="center"><strong>Word Cloud for Keywords</strong></p>

- **Keywords:** A word cloud was generated from the `keyword` column to identify the most common keywords associated with disaster-related tweets. This visualization highlighted terms such as "hostage," "derailment," "flood," "forest fire," and "typhoon." The size of each word in the cloud represents its frequency, with larger words indicating more common keywords in the dataset. This visualization emphasized the variety and prevalence of different disaster types discussed in the tweets, providing a visual representation of the critical themes identified during data exploration.

<p align="center">
  <img src="./assets/images/wordcloud_locations.png" alt="Word Cloud for Locations"/>
</p>

<p align="center"><strong>Word Cloud for Locations</strong></p>

- **Locations:** Another word cloud was created from the `location` column, showcasing the geographical areas most mentioned in disaster-related tweets. Locations like "California", "Texas", and "New York" were prominently featured, indicating areas frequently impacted by the reported events.

  Key observations from the location word cloud:
  - **Prominent Locations:** The most frequently mentioned locations include "USA," "New York," "Canada," "UK," and "Nigeria," indicating high tweet activity related to disasters in these areas.
  - **Global Coverage:** The word cloud shows a wide geographic spread, including locations from various continents such as "Australia," "India," "London," and "California."
  - **Significant Cities and Countries:** Both country names (e.g., "USA," "Canada") and city names (e.g., "New York," "London") appear frequently, highlighting the urban centers often discussed in the context of disasters.
  - **Diverse Mention:** The locations range from specific cities and states to broader regions and countries, reflecting the diverse scope of the dataset in terms of geographic mentions.

<p align="center">
  <img src="./assets/images/wordcloud_texts.png" alt="Word Cloud for Texts"/>
</p>

<p align="center"><strong>Word Cloud for Texts</strong></p>

- **Text:** A word cloud was created from the `text` column to analyze the most frequently occurring words in disaster-related tweets. Interestingly, the most prominent term is "t.co," which appears frequently due to the inclusion of links in tweets. Other significant words include "fire," "people," "suicide," "flood," "police," and "killed," reflecting the critical themes discussed in the dataset. The presence of terms like "Hiroshima," "storm," and "crash" highlights specific disasters and incidents that were heavily discussed. Additionally, words like "via," "amp," and "new" show up frequently, possibly indicating common tweet structures and phrasing.

**Keyword Distribution by Target:** 

<p align="center">
  <img src="./assets/images/keyword_distribution_by_target_top10.png" alt="Keyword Distribution - Top 10"/>
</p>

<p align="center"><strong>Keyword Distribution - Top 10</strong></p>
 
- **Top 10 Keywords:** The top 10 keywords predominantly associated with disaster-related tweets (Target == 1) include "derailment," "wreckage," "outbreak," "debris," and "oil spill." These terms highlight the most frequently discussed disaster events in the dataset, with a strong focus on incidents involving significant damage or threat.

<p align="center">
  <img src="./assets/images/keyword_distribution_by_target_least10.png" alt="Keyword Distribution - Least 10"/>
</p>

<p align="center"><strong>Keyword Distribution - Least 10</strong></p>

- **Least 10 Keywords:** The least mentioned keywords in disaster-related tweets (Target == 1) include "blew up," "threat," "screaming," and "electrocute." These terms, while still relevant to disaster contexts, appear far less frequently in the dataset, indicating they are less commonly associated with major disaster events compared to the top keywords.

**Correlation of Links and Target:**  

<p align="center">
  <img src="./assets/images/link_target_bar_chart.png" alt="Correlation of Links and Target"/>
</p>

<p align="center"><strong>Correlation of Links and Target</strong></p>

- **URLs and Disaster Relevance:** The bar chart illustrates the correlation between the presence of URLs in tweets (`has_urls`) and their relevance to disaster-related content (`target`). The data shows that tweets containing URLs (True) are more likely to be classified as relevant to disasters (Target == 1) compared to those without URLs (False). Specifically, there are 2,172 disaster-related tweets with URLs compared to 1,799 non-disaster-related tweets. In contrast, tweets without URLs are predominantly non-disaster-related, with 2,543 such tweets compared to 1,099 disaster-related tweets. This suggests that disaster-related tweets often include URLs, possibly linking to news articles, videos, or other resources related to the event being discussed.

**Additional Insights from EDA:**
- **Keyword Normalization:** The exploratory data analysis (EDA) uncovered the presence of non-standard characters within certain keywords (e.g., `%20` in "forest%20fire"). To address this, keywords were systematically cleaned and standardized. This process not only enhanced the clarity and consistency of the data but also contributed to improved model accuracy by ensuring that similar keywords were treated uniformly.
- **Geographical Trends:** Analysis of the `location` data revealed that disaster-related tweets were often associated with specific regions that are prone to natural calamities or other emergencies. Identifying these geographical patterns helped in understanding the contextual backdrop of the tweets and aided in refining the model to better account for location-based variations in disaster reporting.
- **URL Presence and Content Relevance:** The EDA also highlighted the correlation between the presence of URLs in tweets and their relevance to disasters. Tweets containing URLs were more likely to be disaster-related, indicating that users often include links to external resources, news articles, or updates when discussing urgent events.

These insights were instrumental in guiding the feature engineering process and refining the overall model strategy. By thoroughly understanding the data's distribution, language patterns, and contextual nuances, the team was able to make informed decisions that significantly enhanced the model's predictive capabilities.


### 3. Model Development

**Model Architecture:**
- **Recurrent Neural Networks (RNN):** The project utilized RNNs, specifically Long Short-Term Memory (LSTM) networks, chosen for their ability to capture the sequential nature of text data. The LSTM model processes sequences of words in tweets, learning context and differentiating between disaster-related and non-disaster-related content.
- **Embedding Layer:** To convert the text data into a format suitable for the LSTM model, an embedding layer was employed. This layer transformed the textual input into dense vectors of fixed size, capturing the semantic meaning of the words.
- **LSTM Layers:** Stacked LSTM layers were used to process the sequences. These layers were designed to retain information from previous words in the sequence, helping the model understand the context and importance of each word within the tweet.
- **Fully Connected Layers:** After the LSTM layers, fully connected layers were added to further refine the learned features and produce the final output.
- **Output Layer:** A single neuron with a sigmoid activation function was used as the output layer to predict the probability of a tweet being disaster-related (Target = 1) or not (Target = 0).
- **Model Training:** The model was trained using binary cross-entropy loss and the Adam optimizer, with early stopping implemented to prevent overfitting.

### 4. Model Evaluation & Results

**Performance Metrics:**  
- **Accuracy:** The LSTM model achieved an accuracy of approximately 81% on the validation set.
- **Precision, Recall, F1-Score:** These metrics were also computed to assess the model's performance across different classes, ensuring that both disaster-related and non-disaster-related tweets were accurately identified.
- **Confusion Matrix:** A confusion matrix was used to visualize the true positive, true negative, false positive, and false negative predictions, providing insights into the model's classification performance.

### 5. Conclusion

**Summary of Findings:**
- The project demonstrated that machine learning models, when combined with effective feature engineering and evaluation techniques, can accurately classify disaster-related tweets. The insights gained from this classification can significantly aid in disaster response efforts.

**Impact:**
- This model has the potential to make a substantial impact on disaster management by providing timely, relevant information to authorities and the public, ultimately contributing to more effective and efficient disaster response and mitigation efforts.


## Acknowledgements

This project was part of the Data Scientist Practicum course at Chulalongkorn University. Special thanks to our instructors and peers for their guidance and support throughout the project.
