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
- **Keywords:** A word cloud was generated from the `keyword` column to identify the most common keywords associated with disaster-related tweets. This visualization highlighted terms such as "earthquake", "fire", and "flood", which frequently appeared in disaster contexts. The word cloud visually emphasized the prevalence and importance of these keywords.
  
  ![Word Cloud for Keywords](./assets/images/wordcloud_keywords.png)

- **Locations:** Another word cloud was created from the `location` column, showcasing the geographical areas most mentioned in disaster-related tweets. Locations like "California", "Texas", and "New York" were prominently featured, indicating areas frequently impacted by the reported events.

  ![Word Cloud for Locations](./images/wordcloud_locations.png)

- **Text Content:** A word cloud for the `text` field of disaster-related tweets provided a broad view of the common terms and phrases used in these tweets. This helped in understanding the context and sentiment expressed by users during disaster events.

  ![Word Cloud for Texts](./images/wordcloud_texts.png)

**Keyword Distribution by Target:**  
- A bar chart was created to display the distribution of keywords, differentiated by the `target` label (1 for disaster-related and 0 for non-disaster-related tweets). This chart provided insights into which keywords were most strongly associated with disaster-related content. For example, keywords like "earthquake", "storm", and "evacuation" had higher frequencies in the disaster-related category, aiding in understanding the significance of specific terms.

  ![Keyword Distribution](./images/keyword_distribution.png)

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
