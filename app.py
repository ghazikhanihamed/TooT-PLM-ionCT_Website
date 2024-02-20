import matplotlib

matplotlib.use("Agg")  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# The rest of your imports
from flask import Flask, render_template, request, jsonify
import torch
import re
from transformers import EsmModel, EsmTokenizer
from settings import settings
from classes.Classifier import CNN
from joblib import load
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# Define the tasks and corresponding models
tasks = {"IC_MP": settings.ESM1B, "IT_MP": settings.ESM1B, "IC_IT": settings.ESM2}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained models
trained_models = {
    "IC_MP": load(f"{settings.FINAL_MODELS_PATH}final_model_IC_MP.joblib"),
    "IT_MP": load(f"{settings.FINAL_MODELS_PATH}final_model_IT_MP.joblib"),
    "IC_IT": CNN([3, 7, 9], [128, 64, 32], 0.27, 1280).to(device),
}
# Load the CNN model state for IC_IT
trained_models["IC_IT"].load_state_dict(
    torch.load(f"{settings.FINAL_MODELS_PATH}final_model_IC_IT.pt", map_location=device)
)
trained_models["IC_IT"].eval()


def generate_chart(probabilities, labels):
    fig, ax = plt.subplots()
    ax.bar(labels, probabilities, color=["blue", "orange"])
    ax.set_ylabel("Probabilities")
    ax.set_title("Prediction Probabilities")

    # Convert plot to PNG image
    png_image = BytesIO()
    plt.savefig(png_image, format="png")
    plt.close(fig)  # Close the figure to free memory

    # Encode PNG image to base64 string
    png_image.seek(0)
    base64_image = base64.b64encode(png_image.getvalue()).decode("utf-8")
    return base64_image


def is_valid_protein_sequence(sequence):
    return re.match("^[ACDEFGHIKLMNPQRSTVWY]+$", sequence) is not None


def load_esm_model(model_info):
    model = EsmModel.from_pretrained(model_info["model"])
    tokenizer = EsmTokenizer.from_pretrained(model_info["model"], do_lower_case=False)
    return model, tokenizer


def process_sequence(sequence, task):
    # Load the appropriate ESM model and tokenizer based on the task
    model_info = tasks[task]
    esm_model, tokenizer = load_esm_model(model_info)

    esm_model = esm_model.to(device)
    esm_model.eval()

    # Preprocess the sequence
    sequence = (
        sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    )

    # Tokenize the sequence for ESM model
    inputs = tokenizer(
        sequence,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate representations using ESM model
    with torch.no_grad():
        outputs = esm_model(**inputs)
        representation = (
            outputs.last_hidden_state
        ) 

    if task == "IC_IT":
        cnn_model = trained_models[task]
        cnn_model.eval()

        with torch.no_grad():
            log_probs = cnn_model(representation)
            # Convert log probabilities to probabilities
            probabilities = torch.exp(log_probs).cpu().numpy().flatten()
            prediction = np.argmax(probabilities)
            label = "Ion Channel" if prediction == 1 else "Ion Transporter"
            labels = ["Ion Transporter", "Ion Channel"]
            chart_image = generate_chart(probabilities, labels)
    else:
        lr_model = trained_models[task]
        # Apply average pooling and ensure conversion to numpy for logistic regression prediction
        pooled_representation = np.mean(representation.cpu().numpy(), axis=1)
        probabilities = lr_model.predict_proba(pooled_representation.reshape(1, -1))[0]
        prediction = lr_model.predict(pooled_representation.reshape(1, -1))[0]
        label = (
            "Non-ionic Membrane Protein"
            if prediction == 0
            else ("Ion Channel" if task == "IC_MP" else "Ion Transporter")
        )
        labels = (
            ["Non-ionic Membrane Protein", "Ion Channel"]
            if task == "IC_MP"
            else ["Non-ionic Membrane Protein", "Ion Transporter"]
        )
        chart_image = generate_chart(probabilities, labels)

    return label, probabilities.tolist(), labels, chart_image


@app.route("/submit_sequence", methods=["POST"])
def submit_sequence():
    sequence = request.form["proteinSequence"]
    task = request.form["classificationType"]

    if not is_valid_protein_sequence(sequence):
        return (
            jsonify(
                {
                    "error": "Invalid protein sequence. Please enter a valid sequence consisting of amino acid letters."
                }
            ),
            400,
        )

    prediction, probabilities, chart_labels, chart_image = process_sequence(
        sequence, task
    )
    return jsonify(
        {
            "prediction": prediction,
            "probabilities": probabilities,
            "chartLabels": chart_labels,
            "chartImage": f"data:image/png;base64,{chart_image}",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
    # app.run(debug=True)
