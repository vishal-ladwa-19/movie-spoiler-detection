from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

app = Flask(__name__)


model = tf.keras.models.load_model('D:\miniproject\myenv\movie-spoiler-detection\movie_spoiler_app\_trained_movie_spoiler_model.h5', custom_objects={'KerasLayer': hub.KerasLayer},compile=False)

@app.route('/')
def home():
    reviews = [
    "This movie was a rollercoaster of emotions, I loved every minute of it.p",
    "The cinematography and background score were absolutely stunning.",
    "I can't believe they killed the main character in the end!",
    "A bit slow in the middle, but overall a solid plot and great acting.",
    "The villain's backstory made him really compelling.",
    "It was shocking when the hero turned out to be the villain all along.",
    "Great performance by the supporting cast as well.",
    "Some jokes felt forced, but kids would love the humor.",
    "A beautiful portrayal of friendship and loyalty.",
    "I didn’t expect the spaceship to explode in the final scene.",
    "Loved the callbacks to the original movie.",
    "The soundtrack is going straight to my playlist.",
    "Character development was strong throughout the film.",
    "Good mix of action and emotion.",
    "The final shot stayed with me long after the movie ended.",   
]

    threshold = 0.15
    predictions = model.predict(reviews)
    labels = [pred[0] > 0.15 for pred in predictions]  
    results = zip(reviews, labels)

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
