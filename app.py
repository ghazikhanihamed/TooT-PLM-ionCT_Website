from flask import Flask, render_template, request, jsonify
import torch
import re
from transformers import EsmModel, EsmTokenizer
from settings import settings
from classes.Classifier import CNN
from joblib import load

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

    # Replace UZOB with X in the sequence
    sequence = (
        sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    )

    # Tokenize the sequence for ESM model
    inputs = tokenizer(
        sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate representations using ESM model
    with torch.no_grad():
        outputs = esm_model(**inputs)
        representation = outputs.last_hidden_state.mean(dim=1)  # Average pooling

    # For IC-IT task, use the CNN model for prediction
    if task == "IC_IT":
        cnn_model = trained_models[task]
        cnn_model.eval()

        with torch.no_grad():
            prediction = cnn_model(representation.unsqueeze(0)).argmax().item()
        label = "Ion Channel" if prediction == 1 else "Ion Transporter"
    else:
        # For IC-MP and IT-MP tasks, use the logistic regression model for prediction
        lr_model = trained_models[task]
        flat_representation = (
            representation.cpu().numpy().flatten()
        )  # Flatten the representation
        prediction = lr_model.predict([flat_representation])
        label = (
            "Non-ionic Membrane Protein"
            if prediction == 0
            else ("Ion Channel" if task == "IC_MP" else "Ion Transporter")
        )

    return label


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

    prediction = process_sequence(sequence, task)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
