# MLOps Group Assignment: Task Distribution Plan
**Course:** PGD AI Program, IIT Jodhpur  
**Project:** End-to-End MLOps Pipeline (Docker · GitHub Actions · Kaggle · W&B)  
**Total Value:** 100 Marks  

---

## Team Roles Overview

To guarantee clean Git commit histories and avoid merge conflicts, work is divided by functional tracks. Every member is responsible for specific files and code assets. **All members will contribute equally to writing their respective sections of the final 4–5 page PDF report.**

| Team Member | Core Project Role | Primary File & Code Ownership |
| :--- | :--- | :--- |
| **Member A** | Infrastructure & DevOps Engineer | `.github/workflows/ci.yml`, `Dockerfile` |
| **Member B** | Data Engineer | `src/data_prep.py`, `id2label.json` |
| **Member C** | Machine Learning Engineer | `Kaggle_Notebook.ipynb` (External Link) |
| **Member D** | Inference Engineer & QA Specialist | `src/inference.py`, `requirements.txt`, `.github/workflows/inference.yml` |

---

## Member A: Infrastructure & DevOps Engineer
**Core Focus:** Git Governance, Containerization, and CI Workflows.

* **Task 1: Git Repository Architecture [10 Marks]**
    * Initialize the public GitHub repository with standard `.gitignore`, `README.md`, and license.
    * Create the `develop` branch and configure branch protection rules on `main` (require at least 1 approval PR before merging).
    * Add Members B, C, and D as Collaborators with Write access. Capture the required settings screenshot.
* **Task 6: Docker Containerization [10 Marks]**
    * Write the production-ready `Dockerfile` using a slim Python base image (`python:3.11-slim`).
    * Configure it to accept `ARG HF_MODEL_NAME` with a sensible default value.
    * Build, tag, and push the final container image to a public registry (Docker Hub or GHCR).
* **Task 7.1: Continuous Integration Setup [5 Marks]**
    * Write and deploy `.github/workflows/ci.yml` to automate code linting (`flake8`) on every push to `develop` or pull request to `main`.
* **Report Contribution:** Write the *Git repository setup* and *Docker design* sections of the final PDF report.

---

## Member B: Data Engineer
**Core Focus:** Data Ingestion, Quality Inspection, Processing, and Label Mapping.

* **Task 2.1 & 2.2: Data Inspection & Preprocessing Pipeline [9 Marks]**
    * Select a publicly available dataset (under 50,000 samples) matching the team's chosen task.
    * Analyze and document its structure, size, and class distributions.
    * Write a reusable Python script (`src/data_prep.py`) that handles modality-specific cleaning (e.g., lowercasing/stripping text, resizing/normalizing images, or handling nulls/scaling tabular data).
* **Task 2.3: Label Encoding & Mapping [6 Marks]**
    * Programmatically generate the `id2label.json` file mapping numerical model outputs to human-readable text labels.
    * Ensure the raw dataset is listed in `.gitignore` so large data files are not committed, pushing only the `.json` mapping file to GitHub.
* **Report Contribution:** Write the *Data cleaning decisions* section of the final PDF report, thoroughly justifying the cleaning and normalization strategies used.

---

## Member C: Machine Learning Engineer (Training)
**Core Focus:** Model Sourcing, Kaggle Notebook Tuning, and Hugging Face Deployment.

* **Task 3: Hugging Face Model Selection [10 Marks]**
    * Research and select a compact, task-appropriate pre-trained model (under 200 MB, e.g., DistilBERT or ResNet-18) from the Hugging Face Hub. 
    * Write the initialization code to load the tokenizer and configuration matching Member B’s `id2label.json`.
* **Task 4: Kaggle Training & Experimentation [25 Marks]**
    * Build and configure the Kaggle training notebook, enabling GPU T4 acceleration.
    * Integrate Kaggle Secrets (`WANDB_API_KEY`, `HF_TOKEN`) to avoid hardcoding credentials.
    * Execute **Version 1** and **Version 2** of training by altering a core hyperparameter (e.g., learning rate or batch size), tracking losses, accuracy, and F1 scores live to W&B.
* **Task 5: Model Deployment & Linking [5 Marks]**
    * Write the script block inside the notebook to push the finalized, best-performing model weights and tokenizer directly to a public Hugging Face profile.
    * Programmatically log the public HF model URL into the W&B run summary.
* **Report Contribution:** Write the *Model selection rationale* (referencing the HF model card) and the *Experiment comparison table* sections for the report.

---

## Member D: Inference Engineer & QA Specialist
**Core Focus:** Inference Scripting, Dependency Pinning, Environment Sync, and Live Verification.

* **Task 7.2 & 7.3: Inference Engine & Secrets Implementation [10 Marks]**
    * Write the production-facing inference code (`src/inference.py`). This script must read input via environment variables (`INPUT_TEXT`), download Member C's fine-tuned weights directly from the Hugging Face Hub, execute predictions, and format clean terminal outputs.
    * Author the manual `.github/workflows/inference.yml` file that triggers on `workflow_dispatch` and hooks into Member A's pipeline securely.
* **Pipeline Verification & Environment Sync [2 Marks]**
    * Create and manage the project's strict `requirements.txt` file, ensuring all package dependencies (`transformers`, `torch`, `scikit-learn`, `wandb`) are rigidly version-pinned to protect Docker builds from breaking.
    * Act as the core QA tester: pull Member A's Docker image locally and execute manual runs to verify the container accurately returns predictions via Member C's model weights.
* **Task 8: Weights & Biases Workspace Management [3 Marks]**
    * Configure the shared Weights & Biases project dashboard to **Public**.
    * Set up the **Runs Comparison Table** within the W&B UI to display metrics side-by-side.
* **Report Contribution:** Write the *Challenges & learnings* section of the PDF report, audit all 5 mandatory links for public accessibility, and compile the final repository README checklist.

---

## Repository Architecture Blueprint
To keep our commit paths clean, we will strictly follow this directory structure:

```text
├── .github/workflows/
│   ├── ci.yml                 <-- Written by Member A
│   └── inference.yml          <-- Written by Member D
├── src/
│   ├── data_prep.py           <-- Written by Member B
│   └── inference.py           <-- Written by Member D
├── id2label.json              <-- Generated by Member B
├── Dockerfile                 <-- Written by Member A
├── requirements.txt           <-- Maintained by Member D
└── README.md                  <-- Compiled by Member D (Contributions from All)
