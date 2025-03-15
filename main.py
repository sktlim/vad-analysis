from utils import audio_stream_from_wav
from vad_models.vad_factory import load_vad


def main():
    audio_source = "../original_dihard3_dataset/third_dihard_challenge_eval/data/flac/DH_EVAL_0001.flac"
    lab_file = "../original_dihard3_dataset/third_dihard_challenge_eval/data/sad/DH_EVAL_0001.lab"

    vad_class = load_vad()
    print(f"Using {type(vad_class)}")
    vad_class.label_file_path = lab_file

    # Simulating Streaming
    for i, audio_chunk in enumerate(audio_stream_from_wav(audio_source)):
        # print(f"Calling {vad_class.process_audio_chunk}")
        vad_class.process_audio_chunk(audio_chunk)

    vad_class.log_vad()


if __name__ == "__main__":
    main()
