import json
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.detection import DetectionAccuracy, DetectionErrorRate
from sklearn.metrics import roc_auc_score


class VADType(str, Enum):
    SILERO = "silero"
    PYANNOTE = "pyannote"


def load_lab_file(lab_file_path):
    lab_annotation = Annotation()
    with open(lab_file_path, "r", encoding="utf-8") as lab_file:
        for line in lab_file:
            start, end, label = line.strip().split()
            if label == "speech":  # Only marking speech segments
                lab_annotation[Segment(float(start), float(end))] = "speech"
    return lab_annotation


def annotation_to_frame_labels(annotation, total_duration, frame_duration):
    num_frames = int(total_duration / frame_duration)
    labels = np.zeros(num_frames)

    for segment in annotation.get_timeline():
        start_frame = int(segment.start / frame_duration)
        end_frame = int(segment.end / frame_duration)
        labels[start_frame:end_frame] = 1

    return labels


@dataclass
class VADBase(ABC):
    chunk_size: float
    overlap: float
    sample_rate: int
    label_file_path: Optional[str] = None

    detection_accuracy: float = 0.0
    detection_error_rate_value: float = 0.0
    missed_detection_rate: float = 0.0
    false_alarm_rate: float = 0.0
    roc_auc: float = 0.0

    vad_result: Annotation = field(default_factory=Annotation)
    vad_segments: deque[Annotation] = deque()
    current_offset: float = 0.0
    frame_duration: float = 0.01
    chunk_logs: List[dict] = field(default_factory=list)  # Stores per-chunk logs

    @abstractmethod
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Annotation:
        pass

    def aggregate_results(self):
        aggregated_timeline = Timeline()
        for vad_result in self.vad_segments:
            aggregated_timeline = aggregated_timeline | vad_result.get_timeline()

        self.vad_result = Annotation()
        for segment in aggregated_timeline.support():
            self.vad_result[segment] = "speech"

    def calcMetrics(self):
        """Computes VAD evaluation metrics if a label file is provided."""
        self.aggregate_results()

        if self.label_file_path is None:
            logging.info("Skipping metric calculation: No label file provided.")
            return

        ground_truth = load_lab_file(self.label_file_path)

        if ground_truth is None:
            logging.info("Skipping metric calculation: Ground truth labels are empty.")
            return

        detection_accuracy = DetectionAccuracy()
        detection_error_rate = DetectionErrorRate()

        self.detection_accuracy = detection_accuracy(ground_truth, self.vad_result)
        self.detection_error_rate_value = detection_error_rate(
            ground_truth, self.vad_result
        )

        # Compute error rate components
        detailed_metrics = detection_error_rate.compute_components(
            ground_truth, self.vad_result
        )
        missed_detection_duration = detailed_metrics["miss"]
        false_alarm_duration = detailed_metrics["false alarm"]
        total_reference_duration = detailed_metrics["total"]

        self.missed_detection_rate = (
            (missed_detection_duration / total_reference_duration)
            if total_reference_duration != 0
            else 0
        )
        self.false_alarm_rate = (
            (false_alarm_duration / total_reference_duration)
            if total_reference_duration != 0
            else 0
        )

        # Compute ROC-AUC
        total_duration = self.vad_result.get_timeline().extent().duration
        ground_truth_labels = annotation_to_frame_labels(
            ground_truth, total_duration, self.frame_duration
        )
        vad_labels = annotation_to_frame_labels(
            self.vad_result, total_duration, self.frame_duration
        )

        if len(ground_truth_labels) == 0 or len(vad_labels) == 0:
            logging.warning(
                "Warning: Empty ground truth or VAD labels. \
                    Skipping ROC-AUC calculation."
            )
            self.roc_auc = float("nan")
        else:
            self.roc_auc = roc_auc_score(ground_truth_labels, vad_labels)

        logging.info(
            "Metrics calculated - Accuracy: %.4f, \
                DER: %.4f, Missed: %.4f, False Alarm: %.4f, ROC AUC: %.4f",
            self.detection_accuracy,
            self.detection_error_rate_value,
            self.missed_detection_rate,
            self.false_alarm_rate,
            self.roc_auc,
        )

    def log_vad(self, output_file_dir):
        """
        Logs overall VAD results, runs metrics if labels exist, \
            and saves structured JSON.
        """

        log_dir = os.path.join(output_file_dir, "logs")
        logging.basicConfig(
            level=logging.INFO,  # Adjust to DEBUG for more details
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"{log_dir}/vad_run.log", mode="w"),  # Log to file
                logging.StreamHandler()
            ],
        )

        if self.label_file_path:
            logging.info("Running VAD metric calculations...")
            self.calcMetrics()
        else:
            logging.info("Empty ground truth or VAD labels")
            self.aggregate_results()

        # Compute overall stats
        total_chunks = len(self.chunk_logs)
        avg_processing_time = (
            sum(chunk["processing_time"] for chunk in self.chunk_logs) / total_chunks
            if total_chunks
            else 0
        )
        total_speech_duration = sum(
            chunk["speech_duration"] for chunk in self.chunk_logs
        )

        overall_stats = {
            "total_chunks_processed": total_chunks,
            "total_runtime": round(
                sum(chunk["processing_time"] for chunk in self.chunk_logs), 4
            ),
            "avg_processing_time_per_chunk": round(avg_processing_time, 4),
            "total_speech_detected": round(total_speech_duration, 2),
            "metrics": {
                "detection_accuracy": round(self.detection_accuracy, 4),
                "detection_error_rate": round(self.detection_error_rate_value, 4),
                "missed_detection_rate": round(self.missed_detection_rate, 4),
                "false_alarm_rate": round(self.false_alarm_rate, 4),
                "roc_auc": (
                    round(self.roc_auc, 4) if not np.isnan(self.roc_auc) else None
                ),
            },
        }

        structured_data = {
            "overall_stats": overall_stats,
            "chunk_logs": self.chunk_logs,
        }

        result_dir = os.path.join(output_file_dir, "results")
        os.makedirs(result_dir, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(result_dir, "vad_results.json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(structured_data, json_file, indent=4)

        # Save and save to LAB format
        lab_path = os.path.join(result_dir, "vad_results.lab")
        with open(lab_path, "w", encoding="utf-8") as lab_file:
            for segment, _, label in self.vad_result.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                label = label if label else "speech"
                lab_file.write(f"{start_time:.3f} {end_time:.3f} {label}\n")

        logging.info("Final structured logs saved: vad_results.json")
