# Bank Customer Churn Prediction 🎉

This is my **first machine learning project**, and I'm super excited to share it! The goal was to predict whether a bank customer will churn (leave the bank) based on their profile and account information, using a classic tabular classification dataset.

I used this project to get hands-on with the full ML workflow: exploring data, cleaning/encoding it, handling class imbalance, and then training and comparing several different classification models — from simple ones like KNN to a neural network built with TensorFlow/Keras.

---

## 📊 Dataset

The dataset (`data.csv`) contains 10,000 bank customer records with the following features:

- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`
- `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- `Exited` — the target variable (1 = customer churned, 0 = customer stayed)

Identifier columns (`RowNumber`, `CustomerId`, `Surname`) were dropped since they carry no predictive value.

## 🔍 Exploratory Data Analysis (EDA)

- Checked dataset shape (10,000 rows × 14 columns) and confirmed there were **no missing values**.
- Encoded `Gender` as a binary feature and one-hot encoded `Geography` (France, Germany, Spain).
- Plotted histograms of every feature split by churn status (`Exited` = 0 vs 1) to visually compare distributions between customers who left and those who stayed.
- Built a **crosstab of Gender vs Exited**, which showed female customers churned at a noticeably higher rate than male customers.
- Compared churn behavior for customers with **above-average estimated salary**, split by gender.
- Compared churn behavior for customers with **above-average balance**, split by gender.
- Plotted the distribution of `EstimatedSalary` for churned vs non-churned customers.

### Key EDA observations
- The dataset is **imbalanced** — far fewer customers churn (`Exited = 1`) than stay.
- Female customers appear to churn at a higher rate than male customers, even when controlling for above-average salary or balance.
- No missing data meant minimal cleaning was needed — most of the effort went into encoding categorical variables and scaling.

## 🛠️ Preprocessing

- Split data into **train (60%) / validation (20%) / test (20%)** sets.
- Scaled features using `StandardScaler`.
- Since the training set was imbalanced (far more "stayed" than "churned" customers), used **`RandomOverSampler`** to balance the classes before training.

## 🤖 Models Used

I trained and compared the following classifiers on the same train/test split:

| Model | Library |
|---|---|
| K-Nearest Neighbors (K=7) | `scikit-learn` |
| Gaussian Naive Bayes | `scikit-learn` |
| Logistic Regression | `scikit-learn` |
| Neural Network (2 hidden layers, Dropout, Adam optimizer) | `TensorFlow / Keras` |
| Decision Tree Classifier | `scikit-learn` |
| Random Forest Classifier (200 trees) | `scikit-learn` |

The neural network architecture:
```
Normalization → Dense(16, relu) → Dropout(0.4) → Dense(16, relu) → Dropout(0.4) → Dense(1, sigmoid)
```
Trained for 100 epochs with `binary_crossentropy` loss and the Adam optimizer.

## 📈 Results

Evaluated on the held-out test set (2,000 samples):

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| KNN (K=7) | 0.74 | 0.42 | 0.66 | 0.51 |
| Gaussian Naive Bayes | 0.74 | 0.41 | 0.64 | 0.50 |
| Logistic Regression | 0.71 | 0.67 | 0.39 | 0.49 |
| Neural Network | **0.83** | 0.57 | 0.58 | 0.58 |
| Decision Tree | 0.80 | 0.51 | 0.46 | 0.48 |
| Random Forest | **0.85** | 0.67 | 0.51 | 0.58 |

🏆 **Random Forest was the best overall performer**, with the highest accuracy (0.85) and the best precision on the churn class (0.67) — meaning when it predicts a customer will churn, it's right two-thirds of the time. The **Neural Network** was close behind (0.83 accuracy) and had the best recall/F1 balance for catching churners.

## 💡 Learnings & Findings

- **Class imbalance matters a lot.** Without oversampling, models were heavily biased toward predicting "not churned," since that's the majority class. `RandomOverSampler` helped models actually learn patterns for the minority (churned) class.
- **Ensemble and non-linear models clearly won here.** Random Forest and the Neural Network both jumped to 83–85% accuracy, well above the simpler models (71–74%). Tree-based ensembles and neural nets could pick up on non-linear relationships between features (like credit score, balance, and age) that KNN, Naive Bayes, and Logistic Regression couldn't capture as well.
- **Accuracy alone can be misleading.** Even among the top models, Random Forest had better precision (0.67) while the Neural Network had a slightly better recall/F1 balance (0.58). Depending on the business goal — catching as many at-risk customers as possible vs. avoiding wasted retention offers on customers who wouldn't have left anyway — the "best" model isn't the same one.
- **Simple models are still a useful baseline.** KNN and Naive Bayes were quick to train and reasonable to start with, even though they were ultimately outperformed by Random Forest and the Neural Network.
- **A single Decision Tree overfits more easily than an ensemble of them.** Random Forest (200 trees) beat the single Decision Tree by 5 points of accuracy and meaningfully better precision/recall — a nice concrete illustration of why ensembling helps.
- **Feature scaling is essential** for distance-based (KNN) and gradient-based (Logistic Regression, Neural Network) models — this project reinforced why `StandardScaler` is a standard preprocessing step.
- **EDA pays off.** Spotting the gender-based churn pattern during EDA gave useful intuition for interpreting model behavior later.
- This was a great first end-to-end project — from raw CSV to six trained models — and it gave me a much better feel for the full ML pipeline, not just the "fit the model" step.

## 🧰 Tech Stack

- Python, pandas, NumPy
- scikit-learn
- imbalanced-learn (`RandomOverSampler`)
- TensorFlow / Keras
- Matplotlib

## 🚀 Next Steps

- Try hyperparameter tuning (e.g., `GridSearchCV`) on Random Forest and the Neural Network to push performance further.
- Look at feature importance (from the Random Forest) to understand which features drive churn the most.
- Experiment with other imbalance-handling techniques like SMOTE, and see if they change which model comes out on top.
- Try an XGBoost or LightGBM model as a stronger gradient-boosted baseline against Random Forest.

---

*This is a learning project — feedback and suggestions are very welcome!*
