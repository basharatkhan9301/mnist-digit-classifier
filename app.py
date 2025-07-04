import gradio as gr
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.uint8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# Train Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Prediction function
def predict_digit(image):
    image = Image.fromarray(image).convert("L").resize((28, 28))
    image_array = np.array(image).reshape(1, -1)
    prediction = rf_clf.predict(image_array)[0]
    return f"Predicted Digit: {prediction}"

# Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(shape=(200, 200), image_mode='L', invert_colors=True, source="canvas"),
    outputs="text",
    title="MNIST Digit Recognizer",
    description="Draw a digit (0–9) and get prediction using Random Forest"
)

interface.launch()
