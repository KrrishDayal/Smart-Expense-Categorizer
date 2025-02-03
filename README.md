# Smart Expense Categorizer 
This is a machine learning project aimed at automatically classifying financial transactions into predefined categories like **Food/Drink**, **Transport**, **Groceries**, **Entertainment**, **Travel**, **Health**, and **Automobile**. Using Natural Language Processing (NLP) and machine learning techniques, the model predicts the category of a given transaction based on its description.

The project uses **TF-IDF Vectorization** for text preprocessing and a **Random Forest Classifier** to categorize the transactions. Although the model has shown improvements in prediction accuracy, there is still room for enhancement, particularly with a larger and more balanced dataset.

## Overview  
Manually categorizing financial transactions is time-consuming and error-prone. This project automates that process by leveraging machine learning to classify transaction descriptions into predefined categories. The model takes the transaction description as input and predicts its appropriate category, which can be used to automate financial tracking, budgeting, or personal finance management.

## Features  
- **Automatic Classification**: Classifies transactions into categories such as **Food/Drink**, **Transport**, **Groceries**, **Entertainment**, **Travel**, **Health**, and **Automobile**.
- **Text Preprocessing**: Uses **TF-IDF Vectorization** to convert transaction descriptions into a numerical format that can be used by the machine learning model.
- **Random Forest Classifier**: Trains a **Random Forest** model with hyperparameter tuning to classify transactions.
- **Performance Evaluation**: Evaluates the model's performance using **accuracy**, **precision**, **recall**, **F1-score**, and a **confusion matrix**.
- **Visualizations**: Generates visualizations such as a **confusion matrix heatmap**, a **bar chart of classification metrics**, and **word clouds** for each transaction category.
- **Prediction on New Data**: The model can predict the categories for new, unseen financial transaction descriptions.

## Results and Model Evaluation  
The model achieved **54% accuracy** on the test set, meaning it correctly predicted the category for 54% of the test samples. While this is a step forward, there is still significant room for improvement. The accuracy could be higher if the dataset were larger and more balanced.

### Classification Report  
The classification report provides a breakdown of **precision**, **recall**, and **F1-score** for each category. The model performs well in predicting **Food/Drink** and **Transport** categories but struggles with **Entertainment**, **Health**, and **Travel**.

### Confusion Matrix  
The confusion matrix visualizes how the model is making predictions. It shows where the model is **misclassifying** transactions and where it is performing well.


## Future Improvements  
- **Increase the dataset size**: Add more samples for each category to improve model performance and generalization.
- **Balance the dataset**: Ensure that each category has roughly an equal number of samples to avoid bias.
- **Experiment with different models**: Try **Logistic Regression** or **Gradient Boosting** for better performance.
- **Hyperparameter tuning**: Further optimize the **Random Forest** parameters using **GridSearchCV**.
- **Advanced text preprocessing**: Use **lemmatization** or **word embeddings** for better feature extraction.

## License  
This project is **open-source** and available under the **MIT License**.  

## Acknowledgments  
Thanks to **scikit-learn**, **pandas**, and **matplotlib** for making machine learning easy! 

## How to Use This  
- **Clone the repo** and install dependencies.
- **Run** the `SmartExpenseCategorizer(1).ipynb` script.
- **Review the results**, **visualizations**, and model performance.

## What's Next?  
Feel free to explore the project and provide suggestions for future improvements. We're actively working on expanding the dataset and refining the model. 
