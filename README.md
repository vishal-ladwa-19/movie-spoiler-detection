🎬 Movie Spoiler Detection
Detect spoilers in movie reviews using a fine-tuned BERT model.

📘 Overview
This project classifies IMDb movie reviews as spoilers or non-spoilers using NLP and deep learning. We use a BERT-based model trained on a Kaggle dataset of labeled reviews.

🛠️ Tech Stack
Python, TensorFlow, Keras, Scikit-learn

Hugging Face Transformers

IMDb Spoiler Dataset (Kaggle)


Install requirements
pip install -r requirements.txt

Train the model
python train_bert_spoiler.py

Evaluate
python evaluate_model.py

📈 Results
Accuracy: ~90%

F1 Score: ~90%

ROC-AUC: ~0.93

🔮 Future Work
Lighter models for mobile

Cross-domain generalization

Real-time spoiler filtering
