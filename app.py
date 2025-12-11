# --------------------------------------------------
# Fart Intelligence + LLM Evolution Factory 🧪
# Colab-ready Notebook / app.py
# --------------------------------------------------

# Install dependencies if running in Colab
# --------------------------------------------------
# !pip install torchaudio soundfile gradio requests tqdm pandas

import os
import io
import random
import time
import uuid
import json
import tempfile
import warnings
from typing import List, Dict, Any

import torch
import torchaudio
import soundfile as sf
import gradio as gr
import requests
from tqdm import tqdm
import pandas as pd # Added for log processing

# --------------------------------------------------
# CONFIGURATION — edit before you run
# --------------------------------------------------
COVALENT_API_KEY = os.getenv("COVALENT_API_KEY", None) # optional
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-YOUR_OPENROUTER_KEY") # Placeholder
HF_API_KEY = os.getenv("HF_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# Fallback for API keys if not set as environment variables for demonstration
if OPENROUTER_API_KEY == "sk-YOUR_OPENROUTER_KEY":
    warnings.warn("OPENROUTER_API_KEY not set. Using a dummy key. LLM calls will fail.")


LLM_PROVIDERS = {
    "openrouter": {
        "enabled": True if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "sk-YOUR_OPENROUTER_KEY" else False,
        "endpoint": "https://api.openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-4o-mini",
        "api_key": OPENROUTER_API_KEY,
        "headers": {"HTTP-Referer": "your-site-url", "X-Title": "your-app-name"}
    },
    # Example: add more providers if keys are set
    # "hf": { "enabled": bool(HF_API_KEY), ... },
    # "groq": { "enabled": bool(GROQ_API_KEY), ... },
}

# Placeholder CNN model path — you must supply your own pretrained model
CNN_MODEL_PATH = "fart_cnn_model.pt" # This will be created by cnn_training.py

# Working directories (temp)
DATA_DIR = "./fart_dataset"
SYN_DIR = os.path.join(DATA_DIR, "synthetic")
REAL_DIR = os.path.join(DATA_DIR, "real")
LOG_FILE = os.path.join("fart-lab/logs", "generation_log.jsonl") # Updated path

os.makedirs(SYN_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Create dummy log file if it doesn't exist for initial run
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        pass # Create an empty file


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def load_cnn_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"[WARN] CNN model not found at '{model_path}'. Using dummy model.")
        return None
    # Instantiate the model class from cnn_training.py before loading state_dict
    try:
        from fart_lab.utils.cnn_training import FartClassifierCNN, LABELS # Assuming fart_lab is in path
        model = FartClassifierCNN(num_classes=len(LABELS))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"CNN model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading CNN model from {model_path}: {e}. Using dummy model.")
        return None

# Ensure fart_lab is in system path for imports
if os.getcwd().endswith('fart-lab'):
    # If currently in fart-lab directory
    project_root = os.getcwd()
else:
    # If in parent directory (e.g. Colab root where fart-lab is a subfolder)
    project_root = os.path.join(os.getcwd(), '')

import sys
if project_root not in sys.path:
    sys.path.append(project_root)

# Attempt to import FartClassifierCNN and LABELS
try:
    from fart_lab.utils.cnn_training import FartClassifierCNN, LABELS as CNN_LABELS
except ImportError as e:
    print(f"Could not import FartClassifierCNN or LABELS from fart_lab.utils.cnn_training: {e}")
    print("Ensure fart-lab/utils is in Python path or cnn_training.py exists.")
    CNN_LABELS = ['wet', 'squeaky', 'bass', 'ghost', 'not_fart'] # Fallback

cnn_model = load_cnn_model(CNN_MODEL_PATH)

def preprocess_audio(wav_bytes: bytes):
    waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
    # Optionally resample or normalize, match CNN_training expected input
    # Assuming 16000 sr and mono channel for CNN
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1: # Convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, 16000 # Return processed waveform and target sample rate

def cnn_predict(waveform, sr) -> dict: # Changed return to dict to include labels
    if cnn_model is None:
        # Dummy fallback: random probabilities for each class
        probs = [random.random() for _ in range(len(CNN_LABELS))]
        total = sum(probs)
        probs = [p / total for p in probs] # Normalize to sum to 1
        return {label: p for label, p in zip(CNN_LABELS, probs)}
    with torch.no_grad():
        # Need to ensure waveform is transformed into MelSpec similar to how the CNN was trained
        n_mels = 128 # Assuming this was used in training
        max_len_samples = 16000 # Assuming 1 sec audio

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=400, hop_length=160)(waveform)
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        mel_spec_db = amplitude_to_db(mel_spectrogram)

        # Pad or truncate to max_len for consistent input size
        if mel_spec_db.shape[2] < max_len_samples // 160: # Approximate frames for 1 sec
             padding = (max_len_samples // 160) - mel_spec_db.shape[2]
             mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding))
        elif mel_spec_db.shape[2] > max_len_samples // 160:
             mel_spec_db = mel_spec_db[:, :, :max_len_samples // 160]

        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        out = cnn_model(mel_spec_db.unsqueeze(0)) # Add batch dimension
        probabilities = torch.softmax(out, dim=1).squeeze(0).tolist()
        return {label: prob for label, prob in zip(CNN_LABELS, probabilities)}

def call_llm(prompt: str, provider: str = "openrouter") -> str:
    cfg = LLM_PROVIDERS.get(provider)
    if cfg is None or not cfg.get("enabled"):
        return f"[LLM {provider} disabled or not configured]"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        **cfg.get("headers", {})
    }
    data = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        resp = requests.post(cfg["endpoint"], json=data, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"[LLM Error from {provider}: {e}]"

def generate_synthetic_fart(kind: str = "wet",
                            duration: float = 1.0,
                            sr: int = 16000) -> (Any, int):
    # This function needs numpy, ensure it's imported correctly in cnn_training.py
    # Or move the numpy dependency here or assume it's installed.
    try:
        import numpy as np
    except ImportError:
        print("Numpy not found. Cannot generate synthetic farts.")
        return np.array([]), sr

    t = np.linspace(0, duration, int(sr * duration), False)
    if kind == "wet":
        fart = 0.3 * np.sin(2 * np.pi * 90 * t) + 0.6 * np.random.normal(0, 0.2, t.shape)
    elif kind == "squeaky":
        fart = 0.2 * np.sin(2 * np.pi * 1000 * t) + 0.05 * np.random.normal(0, 0.1, t.shape)
    elif kind == "bass":
        fart = 0.9 * np.sin(2 * np.pi * 40 * t)
    else:  # generic / noise-based fart
        fart = 0.1 * np.random.normal(0, 0.1, t.shape)
    fart = fart / (max(abs(fart.max()), abs(fart.min())) + 1e-9)
    return fart.astype("float32"), sr

def save_audio(waveform, sr, path):
    # Ensure waveform is 1D for sf.write if it's (1, N)
    if isinstance(waveform, torch.Tensor) and waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    sf.write(path, waveform, sr)

def log_generation(record: Dict[str, Any]):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# --------------------------------------------------
# Evolution / Mutation / Training Loop
# --------------------------------------------------
def mutate_prompt(base_prompt: str) -> str:
    try:
        from fart_lab.utils.evolution import mutate_prompt as evolution_mutate_prompt
        return evolution_mutate_prompt(base_prompt)
    except ImportError:
        warnings.warn("Could not import mutate_prompt from fart_lab.utils.evolution. Using fallback.")
        modifiers = ["wet", "squeaky", "bass-boosted", "silent", "echoey", "short", "long"]
        mod = random.choice(modifiers)
        return base_prompt + f"\nLabel this sound as: {mod}."

def is_adversarial(waveform, sr) -> bool:
    try:
        from fart_lab.utils.adversarial_filter import is_adversarial as filter_is_adversarial
        # waveform needs to be torch.Tensor and sr int as per adversarial_filter.py
        return filter_is_adversarial(waveform, sr)
    except ImportError:
        warnings.warn("Could not import is_adversarial from fart_lab.utils.adversarial_filter. Using fallback.")
        return random.random() < 0.05 # For now random chance to reject (simulate adversarial detection)

def pipeline_evaluate(audio_bytes: bytes) -> Dict[str, Any]:
    waveform, sr = preprocess_audio(audio_bytes)

    # Check for adversarial content first
    if is_adversarial(waveform, sr):
        return {"is_adversarial": True, "message": "Adversarial audio detected. Rejecting."}

    cnn_results = cnn_predict(waveform, sr)
    # Get the highest probability class and its probability
    cnn_pred_label = max(cnn_results, key=cnn_results.get)
    cnn_pred_prob = cnn_results[cnn_pred_label]

    results = []
    for name, cfg in LLM_PROVIDERS.items():
        if not cfg.get("enabled"):
            continue

        # The prompt should ideally guide LLM towards classification based on the CNN's output
        prompt = mutate_prompt(
            f"The provided audio is classified by a CNN with highest probability of {cnn_pred_prob:.2f} as '{cnn_pred_label}'. "
            f"Please analyze this sound. Do you agree with the CNN's classification? If not, why? "
            f"Describe its characteristics and provide your own classification."
        )
        resp = call_llm(prompt, name)

        # Dummy fitness for now; in a real scenario, this would involve
        # `calculate_llm_fitness` from utils.evolution based on a known expected label
        # For this demo, let's assume LLM tries to agree with CNN if it's confident.
        fitness = random.random() # Placeholder
        # A more advanced fitness would compare LLM's classification to cnn_pred_label or a true label.

        results.append({"provider": name, "prompt": prompt, "response": resp, "fitness": fitness})

    best = max(results, key=lambda x: x["fitness"]) if results else None
    return {"is_adversarial": False, "cnn_predictions": cnn_results, "best_llm_analysis": best, "all_llm_results": results}

# --------------------------------------------------
# Gradio / UI Interface
# --------------------------------------------------
def classify_and_show(audio_file):
    if audio_file is None:
        return "Please upload an audio file."

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    out = pipeline_evaluate(audio_bytes)

    if out.get("is_adversarial"):
        return f"Adversarial Detection: {out['message']}"

    cnn_preds = out['cnn_predictions']
    cnn_text = "CNN Predictions:\n" + "\n".join([f"  - {label}: {prob:.2f}" for label, prob in cnn_preds.items()]) + "\n\n"

    text = cnn_text

    if out.get("best_llm_analysis"):
        text += "=== LLM Best Analysis ===\n"
        text += f"Provider: {out['best_llm_analysis']['provider']}\n"
        text += f"Fitness: {out['best_llm_analysis']['fitness']:.2f}\n"
        text += "Response:\n" + out["best_llm_analysis"]["response"]
    else:
        text += "No LLM analysis available (API key not configured or error)."
    return text

# --- Dashboard Functions ---
def _load_llm_logs_df() -> pd.DataFrame:
    logs = []
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return pd.DataFrame()

    if not logs:
        return pd.DataFrame()

    df = pd.DataFrame(logs)
    # Filter for LLM related logs if the log file contains other types
    if 'type' in df.columns and 'llm_training_data_candidate' in df['type'].values:
        df = df[df['type'] == 'llm_training_data_candidate']
    return df

def load_llm_logs() -> str:
    df = _load_llm_logs_df()
    if df.empty:
        return "<p>No LLM logs found or log file is empty.</p>"
    return df.to_html(index=False)


def get_cnn_performance_metrics() -> str:
    # This function would ideally read from a CNN training log file or a dedicated metrics JSON
    # For now, return placeholder metrics.
    metrics = {
        "Accuracy": f"{random.uniform(0.75, 0.95):.2f}",
        "Precision": f"{random.uniform(0.70, 0.90):.2f}",
        "Recall": f"{random.uniform(0.70, 0.90):.2f}",
        "F1-Score": f"{random.uniform(0.70, 0.90):.2f}",
        "Last Trained": "2024-07-20",
        "Dataset Size": f"{random.randint(1000, 5000)} samples"
    }
    return "### CNN Performance Metrics (Placeholder)\n" + "\n".join([f"- {k}: {v}" for k, v in metrics.items()])

def get_fitness_score_trends() -> str:
    # This would involve parsing log data over time and potentially plotting.
    # For now, return a placeholder describing trends.
    trends = [
        "Overall LLM Fitness: steadily increasing (e.g., +5% last week)",
        "CNN Model Loss: slowly decreasing",
        "Synthetic Data Generation Rate: ~100 samples/day",
        "Adversarial Rejection Rate: ~5%"
    ]
    return "### Fitness Score Trends (Placeholder)\n" + "\n".join([f"- {t}" for t in trends])

def get_llm_leaderboard() -> str:
    logs_df = _load_llm_logs_df()
    if logs_df.empty or 'fitness_score' not in logs_df.columns or 'provider' not in logs_df.columns:
        return "<p>Not enough data to generate leaderboard.</p>"

    # Ensure fitness_score is numeric for calculation
    logs_df['fitness_score'] = pd.to_numeric(logs_df['fitness_score'], errors='coerce')
    logs_df.dropna(subset=['fitness_score'], inplace=True)

    if logs_df.empty:
        return "<p>Not enough data to generate leaderboard after cleaning.</p>"

    leaderboard = logs_df.groupby('provider')['fitness_score'].mean().reset_index()
    leaderboard = leaderboard.sort_values(by='fitness_score', ascending=False)
    leaderboard['fitness_score'] = leaderboard['fitness_score'].apply(lambda x: f"{x:.3f}")
    leaderboard.rename(columns={'fitness_score': 'Average Fitness Score'}, inplace=True)

    return leaderboard.to_html(index=False)


# --------------------------------------------------
# Main Entrypoints
# --------------------------------------------------
if __name__ == "__main__":
    print("Starting Fart Intelligence UI...")

    # Create the main audio interface
    audio_interface = gr.Interface(
        fn=classify_and_show,
        inputs=gr.Audio(sources=["upload"], type="filepath", label="Upload Fart Audio"),
        outputs=gr.Markdown(label="Analysis Result"),
        title="Fart Intelligence: Audio Analysis",
        description="Upload a sound clip for CNN classification and LLM reasoning."
    )

    # Create the LLM Logs interface
    llm_logs_interface = gr.Interface(
        fn=load_llm_logs,
        inputs=[],
        outputs=gr.HTML(label="LLM Generation Logs"), # Use gr.HTML for DataFrame.to_html
        title="LLM Generation Logs",
        description="Raw logs from LLM prompt mutations and responses."
    )

    # Create the CNN Metrics interface
    cnn_metrics_interface = gr.Interface(
        fn=get_cnn_performance_metrics,
        inputs=[],
        outputs=gr.Markdown(label="CNN Performance"),
        title="CNN Performance Metrics",
        description="Overview of the CNN model's current performance."
    )

    # Create the Fitness Trends interface
    fitness_trends_interface = gr.Interface(
        fn=get_fitness_score_trends,
        inputs=[],
        outputs=gr.Markdown(label="Evolutionary Trends"),
        title="Fitness Score Trends",
        description="Trends in LLM and CNN fitness over time."
    )

    # Create the LLM Leaderboard interface
    llm_leaderboard_interface = gr.Interface(
        fn=get_llm_leaderboard,
        inputs=[],
        outputs=gr.HTML(label="LLM Leaderboard"), # Use gr.HTML for DataFrame.to_html
        title="LLM Leaderboard",
        description="Ranking of LLMs based on average fitness scores."
    )

    # Combine interfaces into a TabbedInterface
    demo = gr.TabbedInterface(
        [audio_interface, llm_logs_interface, cnn_metrics_interface, fitness_trends_interface, llm_leaderboard_interface],
        ["Audio Analysis", "LLM Logs", "CNN Metrics", "Fitness Trends", "LLM Leaderboard"]
    )

    demo.launch(share=True, debug=True)
