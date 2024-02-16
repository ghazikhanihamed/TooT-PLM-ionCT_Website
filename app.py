from flask import Flask, render_template, request, jsonify
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import torch
from joblib import load
import numpy as np
from settings import settings
from transformers import EsmModel, EsmTokenizer
from classes.Classifier import CNN
from torch.utils.data import TensorDataset, DataLoader
from classes.PLMDataset import GridDataset

app = Flask(__name__)

# Define the tasks and corresponding models
tasks = {"IC_MP": settings.ESM1B, "IT_MP": settings.ESM1B, "IC_IT": settings.ESM2}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained models
trained_models = {
    "IC_MP": load(f"{settings.FINAL_MODELS_PATH}/final_model_IC_MP.joblib"),
    "IT_MP": load(f"{settings.FINAL_MODELS_PATH}/final_model_IT_MP.joblib"),
    "IC_IT": CNN([3, 7, 9], [128, 64, 32], 0.27, 1280).to(device),
}
# Load the CNN model state for IC_IT
trained_models["IC_IT"].load_state_dict(
    torch.load(f"{settings.FINAL_MODELS_PATH}/final_model_IC_IT.pt")
)
trained_models["IC_IT"].eval()


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

        # Pass the representation through the CNN model
        with torch.no_grad():
            prediction = cnn_model(representation.unsqueeze(0)).argmax().item()
        label = "IC" if prediction == 1 else "IT"
    else:
        # For IC-MP and IT-MP tasks, use the logistic regression model for prediction
        lr_model = trained_models[task]
        flat_representation = (
            representation.cpu().numpy().flatten()
        )  # Flatten the representation
        prediction = lr_model.predict([flat_representation])
        label = "MP" if prediction == 0 else ("IC" if task == "IC_MP" else "IT")

    return label


def generate_representation(sequence, model, tokenizer):
    # Replace UZOB with X
    sequence = (
        sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    )

    # Tokenize the sequence
    inputs = tokenizer(
        sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate representations
    with torch.no_grad():
        outputs = model(**inputs)
        representation = (
            outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        )  # Average pooling

    return representation


def make_prediction(representation, task):
    model = trained_models[task]

    # For IC-IT, use the CNN model
    if task == "IC_IT":
        prediction = model(torch.tensor(representation).unsqueeze(0)).argmax().item()
    else:  # For IC-MP and IT-MP, use Logistic Regression
        prediction = model.predict(representation)

    # Convert prediction to label
    if task == "IC_IT":
        label = "IC" if prediction == 1 else "IT"
    elif task == "IC_MP":
        label = "MP" if prediction == 0 else "IC"
    else:
        label = "MP" if prediction == 0 else "IT"

    return label


@app.route("/submit_sequence", methods=["POST"])
def submit_sequence():
    sequence = request.form["proteinSequence"]
    task = request.form["task"]

    prediction = process_sequence(sequence, task)
    return jsonify({"sequence": sequence, "prediction": prediction})


@app.route("/submit_file", methods=["POST"])
def submit_file():
    if "fastaFile" in request.files:
        fasta_file = request.files["fastaFile"]
        task = request.form["task"]
        predictions = {}

        if task == "IC_IT":
            sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
            model_info = tasks[task]
            esm_model, tokenizer = load_esm_model(model_info)
            esm_model = esm_model.to(device)
            esm_model.eval()

            representations = []
            for sequence in sequences:
                # Replace UZOB with X
                sequence = (
                    sequence.replace("U", "X")
                    .replace("Z", "X")
                    .replace("O", "X")
                    .replace("B", "X")
                )

                # Tokenize the sequence
                inputs = tokenizer(
                    sequence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate representations
                with torch.no_grad():
                    outputs = esm_model(**inputs)
                    representation = outputs.last_hidden_state.mean(
                        dim=1
                    )  # Average pooling
                    representations.append(representation.cpu().numpy())

            # Convert list of numpy arrays to a single numpy array
            representations = np.array(representations)

            # Since we don't have labels for prediction, create dummy labels
            dummy_labels = np.zeros(len(representations))

            # Create dataset and dataloader for CNN model
            test_dataset = GridDataset(
                torch.tensor(representations).float(), torch.tensor(dummy_labels)
            )
            test_loader = DataLoader(
                test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False
            )

            cnn_model = trained_models[task]
            cnn_model.eval()

            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(device)
                    outputs = cnn_model(data)
                    predictions_batch = outputs.argmax(dim=1).cpu().numpy()
                    for i, pred in enumerate(predictions_batch):
                        predictions[sequences[i]] = "IC" if pred == 1 else "IT"
        else:
            for record in SeqIO.parse(fasta_file, "fasta"):
                prediction = process_sequence(str(record.seq), task)
                predictions[record.id] = prediction

        return jsonify(predictions)

    return jsonify({"error": "No file found"})


if __name__ == "__main__":
    app.run(debug=True)
