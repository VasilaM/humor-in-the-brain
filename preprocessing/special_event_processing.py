import os
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub


# -----------------------------
# CONFIG
# -----------------------------
JSON_DIR = "my_transcripts"          # directory containing existing transcript JSON files
OUTPUT_DIR = "transcripts_with_events"  # where updated JSON files will be saved

# Detection settings
TARGET_LABELS = {
    "Laughter": "audience_laughter",
    "Applause": "applause",
    "Clapping": "applause",
}

CONFIDENCE_THRESHOLDS = {
    "Laughter": 0.20,
    "Applause": 0.15,
    "Clapping": 0.15,
}

MIN_EVENT_DURATION_SEC = 0.40
MERGE_GAP_SEC = 0.75

# YAMNet parameters
YAMNET_SAMPLE_RATE = 16000
PATCH_HOP_SECONDS = 0.48  # YAMNet prediction stride is ~0.48s


# -----------------------------
# YAMNet setup
# -----------------------------
print("Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

print("Loading YAMNet class map...")
class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
class_names = pd.read_csv(class_map_path)["display_name"].tolist()


# -----------------------------
# Helpers
# -----------------------------
def load_audio_mono_16k(audio_path: str):
    """Load audio as mono float32 at 16kHz."""
    waveform, sr = librosa.load(audio_path, sr=YAMNET_SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32), sr


def get_target_class_indices():
    """
    Find all YAMNet class indices matching the labels we care about.
    This uses exact class names in TARGET_LABELS.
    """
    label_to_idx = {}
    for i, name in enumerate(class_names):
        if name in TARGET_LABELS:
            label_to_idx[name] = i
    return label_to_idx


TARGET_CLASS_INDICES = get_target_class_indices()

if not TARGET_CLASS_INDICES:
    raise RuntimeError(
        "Could not find target classes in YAMNet class map. "
        "Check TARGET_LABELS keys against the model's class names."
    )


def detect_target_events(audio_path: str):
    """
    Run YAMNet and return raw frame-level detections for laughter/applause-like classes.
    Each detection is a dict with label, start, end, confidence.
    """
    waveform, sr = load_audio_mono_16k(audio_path)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores = scores.numpy()  # shape: (num_patches, num_classes)

    events = []
    num_frames = scores.shape[0]

    for frame_idx in range(num_frames):
        frame_start = frame_idx * PATCH_HOP_SECONDS
        frame_end = frame_start + PATCH_HOP_SECONDS

        for class_name, class_idx in TARGET_CLASS_INDICES.items():
            confidence = float(scores[frame_idx, class_idx])
            threshold = CONFIDENCE_THRESHOLDS.get(class_name, 0.2)

            if confidence >= threshold:
                events.append({
                    "raw_label": class_name,
                    "label": TARGET_LABELS[class_name],
                    "start": frame_start,
                    "end": frame_end,
                    "confidence": confidence,
                })

    return events


def merge_events(events, merge_gap_sec=0.75, min_duration_sec=0.40):
    """
    Merge nearby events of the same label and keep max confidence.
    """
    if not events:
        return []

    events = sorted(events, key=lambda x: (x["label"], x["start"]))
    merged = []

    current = events[0].copy()

    for evt in events[1:]:
        same_label = evt["label"] == current["label"]
        close_in_time = evt["start"] <= current["end"] + merge_gap_sec

        if same_label and close_in_time:
            current["end"] = max(current["end"], evt["end"])
            current["confidence"] = max(current["confidence"], evt["confidence"])
        else:
            if (current["end"] - current["start"]) >= min_duration_sec:
                merged.append(current)
            current = evt.copy()

    if (current["end"] - current["start"]) >= min_duration_sec:
        merged.append(current)

    return merged


def make_event_id(prefix="evt"):
    """Create a lightweight unique event id."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def append_events_to_json(data: dict, new_events: list):
    """
    Append new detected events into data['annotations']['events'].
    """
    data.setdefault("annotations", {})
    data["annotations"].setdefault("events", [])

    existing_events = data["annotations"]["events"]

    for evt in new_events:
        existing_events.append({
            "event_id": make_event_id(),
            "label": evt["label"],
            "start": round(float(evt["start"]), 2),
            "end": round(float(evt["end"]), 2),
            "annotator": "yamnet_auto",
            "notes": (
                f"Auto-detected from YAMNet "
                f"(source class={evt['raw_label']}, confidence={evt['confidence']:.3f})"
            ),
        })

    # Sort by start time after appending
    existing_events.sort(key=lambda x: (x["start"], x["end"]))
    return data


def update_json_file(json_path: Path, output_dir: Path):
    """
    Read one JSON, detect events from its audio file, append them, save updated JSON.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_path = data.get("audio_filepath")
    if not audio_path:
        raise ValueError(f"{json_path} has no 'audio_filepath' field.")

    if not os.path.isabs(audio_path):
        audio_path = os.path.join(audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    raw_events = detect_target_events(audio_path)
    merged_events = merge_events(
        raw_events,
        merge_gap_sec=MERGE_GAP_SEC,
        min_duration_sec=MIN_EVENT_DURATION_SEC,
    )

    updated_data = append_events_to_json(data, merged_events)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / json_path.name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    return {
        "json_file": str(json_path),
        "audio_file": audio_path,
        "num_raw_events": len(raw_events),
        "num_merged_events": len(merged_events),
        "output_file": str(output_path),
    }


def main():
    json_dir = Path(JSON_DIR)
    output_dir = Path(OUTPUT_DIR)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir.resolve()}")
        return

    summaries = []
    for json_file in json_files:
        try:
            summary = update_json_file(json_file, output_dir)
            summaries.append(summary)
            print(
                f"Updated {json_file.name}: "
                f"{summary['num_merged_events']} merged events "
                f"-> {summary['output_file']}"
            )
        except Exception as e:
            print(f"Failed on {json_file.name}: {e}")

    print("\nDone.")
    print("Summary:")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()