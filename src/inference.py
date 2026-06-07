import os
import wandb
from transformers import pipeline

MODEL_ID = "mehtayash12345678/mlops-ag_news_classification-distilbert"

WANDB_PROJECT = "mlops-ag_news_classification-distilbert"
WANDB_ENTITY = "g25ait2133-indian-institute-technology-jodhpur"


def main():
    input_text = os.environ.get("INPUT_TEXT")
    if not input_text:
        raise ValueError("INPUT_TEXT environment variable is not set.")

    hf_token = os.environ.get("HF_TOKEN")

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="inference",
    )

    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=hf_token,
    )

    result = classifier(input_text)[0]
    label_name = result["label"]
    confidence = result["score"]

    print(f"Input: {input_text}")
    print(f"Predicted Label: {label_name}")
    print(f"Confidence: {confidence:.4f}")

    # Write to GitHub Actions Job Summary
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write("## Inference Result\n\n")
            f.write(f"**Input:** {input_text}\n\n")
            f.write(f"| Predicted Label | Confidence |\n")
            f.write(f"|---|---|\n")
            f.write(f"| **{label_name}** | `{confidence:.4f}` |\n")

    wandb.log(
        {
            "input_text": input_text,
            "predicted_label": label_name,
            "confidence": confidence,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
