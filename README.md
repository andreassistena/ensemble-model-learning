# ensemble-model-learning
📩 SMS Spam Detection with Naive Bayes & Ensemble Methods
This project applies various supervised learning algorithms to detect spam messages using natural language processing techniques and ensemble machine learning models.

🧠 Objective
Classify SMS messages as spam or ham using the Naive Bayes classifier, and then evaluate performance improvements using three powerful ensemble methods:

Bagging Classifier

Random Forest Classifier

AdaBoost Classifier

📁 Dataset
The dataset used is the SMS Spam Collection, which contains 5,574 English messages labeled as spam or ham.

Each message is labeled as:

spam – unwanted/unsolicited messages

ham – legitimate messages

🔧 Tools & Libraries
Python

Pandas

scikit-learn (sklearn)

Natural Language Processing (CountVectorizer)

⚙️ Workflow
Preprocessing

Read and clean the dataset

Vectorize the text using CountVectorizer

Modeling (Naive Bayes)

Train a Multinomial Naive Bayes classifier

Evaluate using accuracy, precision, recall, and F1 score

Ensemble Methods

Implement and train:

BaggingClassifier (200 estimators)

RandomForestClassifier (200 estimators)

AdaBoostClassifier (300 estimators, learning rate = 0.2)

Compare results to Naive Bayes baseline

📊 Results
Each model is evaluated using the following metrics:

Accuracy – overall correctness

Precision – how many predicted spams were actually spam

Recall – how many actual spams were correctly predicted

F1 Score – balance between precision and recall

python
Copy
Edit
def print_metrics(y_true, preds, model_name=None):
    ...
✅ Example Output
(Insert your actual metrics here after running)

Model	Accuracy	Precision	Recall	F1 Score
Naive Bayes	0.9885	0.9721	0.9405	0.9560
Bagging	...	...	...	...
Random Forest	...	...	...	...
AdaBoost	...	...	...	...

🧭 Future Work
Try TfidfVectorizer for text feature extraction

Use GridSearchCV for hyperparameter tuning

Deploy the best model as an API

