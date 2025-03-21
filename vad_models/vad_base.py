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

from utils import setup_logger


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
    output_dir: Optional[str] = None
    filename: Optional[str] = None

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

    def log_vad(self):
        """
        Logs overall VAD results, runs metrics if labels exist, \
            and saves structured JSON.
        """

        vad_dir = os.path.join(self.output_dir, self.filename)
        os.makedirs(vad_dir, exist_ok=True)

        # Setup loggers
        run_logger = setup_logger(vad_dir, "", "run_log")

        if self.label_file_path:
            run_logger.info("Running VAD metric calculations...")
            self.calcMetrics()
        else:
            run_logger.info("No label file provided. Skipping metric calculations.")

        # Compute final structured results
        structured_data = {
            "overall_stats": {
                "total_chunks_processed": len(self.chunk_logs),
                "total_runtime": sum(
                    chunk["processing_time"] for chunk in self.chunk_logs
                ),
                "avg_processing_time_per_chunk": sum(
                    chunk["processing_time"] for chunk in self.chunk_logs
                )
                / max(len(self.chunk_logs), 1),
                "total_speech_detected": sum(
                    chunk["speech_duration"] for chunk in self.chunk_logs
                ),
                "metrics": {
                    "detection_accuracy": self.detection_accuracy,
                    "detection_error_rate": self.detection_error_rate_value,
                    "missed_detection_rate": self.missed_detection_rate,
                    "false_alarm_rate": self.false_alarm_rate,
                    "roc_auc": self.roc_auc if not np.isnan(self.roc_auc) else None,
                },
            },
            "chunk_logs": self.chunk_logs,
        }

        # Save results as JSON
        json_path = os.path.join(vad_dir, f"{self.filename}_results.json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(structured_data, json_file, indent=4)

        run_logger.info(f"Final structured logs saved: {json_path}")

        # Save as .lab file
        lab_path = os.path.join(vad_dir, f"{self.filename}.lab")
        with open(lab_path, "w", encoding="utf-8") as lab_file:
            for segment, _, label in self.vad_result.itertracks(yield_label=True):
                start_time = segment.start
                end_time = segment.end
                label = label if label else "speech"
                lab_file.write(f"{start_time:.3f} {end_time:.3f} {label}\n")

        run_logger.info(f"VAD lab results saved: {lab_path}")
    
    def reset(self):
        self.vad_segments.clear()
        self.vad_result = Annotation()
        self.current_offset = 0.0
        self.chunk_logs.clear()

        self.detection_accuracy = 0.0
        self.detection_error_rate_value = 0.0
        self.missed_detection_rate = 0.0
        self.false_alarm_rate = 0.0
        self.roc_auc = 0.0
