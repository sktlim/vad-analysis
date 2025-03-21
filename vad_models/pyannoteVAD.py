from typing import Optional

import numpy as np
import torch
import yaml
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation, Segment

from utils import time_it
from vad_models.vad_base import VADBase


class PyannoteStreamingVAD(VADBase):
    def __init__(
        self,
        chunk_size: float,
        overlap: float,
        sample_rate: int,
        label_file_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        super().__init__(chunk_size, overlap, sample_rate, label_file_path, output_dir, filename)

        with open("configs/vad_config.yaml", "r", encoding="UTF-8") as file:
            config = yaml.safe_load(file)
        AUTH_TOKEN = config.get("AUTH_TOKEN")

        self.model = Model.from_pretrained(
            "pyannote/segmentation-3.0", use_auth_token=AUTH_TOKEN
        )
        self.pipeline = VoiceActivityDetection(segmentation=self.model)
        self.pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    @time_it
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Annotation:
        """
        Processes a single audio chunk, returns the VAD result on the fly, and \
            updates the global aggregation.
        """
        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)

        input_dict = {"waveform": audio_tensor, "sample_rate": self.sample_rate}
        vad_result = self.pipeline(input_dict)

        adjusted_annotation = Annotation()
        for segment, _, label in vad_result.itertracks(yield_label=True):
            adjusted_segment = Segment(
                segment.start + self.current_offset, segment.end + self.current_offset
            )
            adjusted_annotation[adjusted_segment] = label

        self.vad_segments.append(adjusted_annotation)
        self.current_offset += self.chunk_size - self.overlap

        return adjusted_annotation
