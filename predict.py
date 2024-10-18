import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
import uuid
from typing import List
import nltk
import numpy as np
import requests
import torch
import whisperx
from cog import BasePredictor, BaseModel, Input
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from pydub import AudioSegment
from speechbrain.pretrained import EncoderClassifier
from transcription_helpers import transcribe_batched
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
import argparse
import logging
import os
import re
import torchaudio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from transcription_helpers import transcribe_batched


# Configure logging
logging.basicConfig(level=logging.INFO)

# Define language lists and model types
punct_model_langs = [
    "en", "fr", "de", "es", "it", "nl", "pt",
    "bg", "pl", "cs", "sk", "sl",
]
wav2vec2_langs = list(DEFAULT_ALIGN_MODELS_TORCH.keys()) + list(DEFAULT_ALIGN_MODELS_HF.keys())
whisper_langs = sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()])
mtypes = {"cpu": "int8", "cuda": "float16"}


class Output(BaseModel):
    segments: List[dict]


def send_pubsub_message(project_id, topic_id, message_dict, credentials):
    """Sends a message to Google Pub/Sub."""
    try:
        # Decode the base64 encoded credentials
        decoded_credentials = base64.b64decode(credentials).decode('utf-8')

        # Load the JSON credentials
        credentials_info = json.loads(decoded_credentials, strict=False)

        # Create credentials object
        credentials_obj = service_account.Credentials.from_service_account_info(credentials_info)

        # Create Pub/Sub publisher client
        publisher = pubsub_v1.PublisherClient(credentials=credentials_obj)
        topic_path = publisher.topic_path(project_id, topic_id)

        # Publish the message
        future = publisher.publish(topic_path, json.dumps(message_dict).encode('utf-8'))
        future.result()  # Verify that the message was published successfully

    except Exception as e:
        logging.error(f"Failed to send message to Pub/Sub: {e}")
        traceback.print_exc()


def get_audio_segment(signal, start_time, end_time):
    """Extracts a segment of the audio signal between start_time and end_time."""
    return signal[int(start_time * 1000):int(end_time * 1000)]  # Convert seconds to milliseconds

def get_sentences_speaker_mapping(transcription_base, sentences, audio):
    """
    Processes the list of words with speaker labels and groups them into sentences
    with speaker embeddings.

    Args:
        sentences (list): List of dictionaries containing words with start_time, end_time, word, speaker.
        audio (AudioSegment): AudioSegment object of the audio.

    Returns:
        list: List of sentences with speaker embeddings.
    """
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_speechbrain"
    )
    # Extract speaker embeddings
    for segment in sentences:
        try:
            audio_segment = get_audio_segment(audio, segment["start"], segment["end"])
            # Convert audio segment to numpy array
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            # Normalize samples
            max_abs_value = float(1 << (8 * audio_segment.sample_width - 1))
            samples = samples / max_abs_value
            # Convert to tensor
            audio_tensor = torch.from_numpy(samples).unsqueeze(0)
            # Compute embeddings
            wav_lens = torch.tensor([1.0])
            embeddings = classifier.encode_batch(audio_tensor, wav_lens)
            # Save embeddings
            embeddings_np = embeddings.squeeze().detach().cpu().numpy()
            segment["speaker_embedding"] = embeddings_np.tolist()  # Convert to list for JSON serialization
        except:
            pass

    return sentences


class Predictor(BasePredictor):
    def setup(self):
        """Load necessary models and configurations."""
        nltk.download('punkt')

    def predict(
        self,
        file_url: str = Input(
            description="A direct audio file URL", default=None
        ),
        language: str = Input(
            description="Language spoken in the audio, specify None to perform language detection",
            default="es"
        ),
        batch_size: int = Input(
            description="Batch size for batched inference",
            default=8
        ),
        multimedia_id: str = Input(
            description="Multimedia ID", default=None
        ),
        project_id: str = Input(
            description="GCP Project ID for Pub/Sub", default=None
        ),
        topic_id: str = Input(
            description="Pub/Sub Topic ID", default=None
        ),
        credentials: str = Input(
            description="GCP Service Account Credentials", default=None
        ),
        hf_token: str = Input(
            description="HuggingFace token", default="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"
        ),
        min_num_speakers: int = Input(
            description="Min number of speakers", default=None
        ),
        max_num_speakers: int = Input(
            description="Max number of speakers", default=None
        )
    ) -> Output:
        if file_url is None:
            raise ValueError("ERROR: 'file_url' is required!")

        random_uuid = uuid.uuid4()
        vocal_target  = f"temp-{random_uuid}.wav"
        temp_outputs_dir = f"temp_{random_uuid}_outputs"

        try:
            # Download and convert audio to WAV
            vocal_target = self.download_audio_and_convert_to_wav(file_url,vocal_target)

            # Separate vocals using Demucs
            self.separate_vocals(vocal_target, temp_outputs_dir)

            # Update vocal_target to point to the separated vocals
            vocal_target = os.path.join(
                temp_outputs_dir,
                "htdemucs",
                os.path.splitext(os.path.basename(vocal_target))[0],
                "vocals.wav",
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Transcribe using WhisperX
            segments,language = transcribe_batched(
                vocal_target,
                language,
                batch_size,
                "large-v2",
                mtypes[device],
                device,
            )

            if language in wav2vec2_langs:
                # 2. Align whisper output
                model_a, metadata = whisperx.load_align_model(
                    language_code=language, device=device
                )
                audio = whisperx.load_audio(vocal_target)
                results = whisperx.align(
                    segments,
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False
                )
                transcrition_base = results
                # 3. Assign speaker labels
                if hf_token:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=hf_token, device=device
                    )
                else:
                    diarize_model = whisperx.DiarizationPipeline(device=device)
                # TODO: Add min/max number of speakers if known
                diarize_segments = diarize_model(audio)
                results = whisperx.assign_word_speakers(diarize_segments, results)
                # Clear GPU VRAM
                del diarize_model
                torch.cuda.empty_cache()

                logging.info(f"segments: {results}")

                # Get sentences with speaker mapping
                segments = get_sentences_speaker_mapping(
                    results["segments"],
                    AudioSegment.from_file(vocal_target).set_channels(1)
                )

                # Send success message to Pub/Sub if credentials are provided
                if credentials and project_id and topic_id and multimedia_id:
                    send_pubsub_message(
                        project_id,
                        topic_id,
                        {"id": multimedia_id, "status": "success"},
                        credentials
                    )

                return Output(segments=segments)

            else:
                # Handle case where language is not supported
                raise ValueError(f"Language '{language}' is not supported for alignment.")

        except Exception as e:
            # Send failure message to Pub/Sub if credentials are provided
            if credentials and project_id and topic_id and multimedia_id:
                send_pubsub_message(
                    project_id,
                    topic_id,
                    {"id": multimedia_id, "status": "failed", "error": str(e)},
                    credentials
                )
            logging.error(f"Error running inference: {e}")
            traceback.print_exc()
            raise
        finally:
            # Clean up temporary files and directories
            try:
                if 'vocal_target' in locals() and os.path.exists(vocal_target):
                    os.remove(vocal_target)
                if os.path.exists(temp_outputs_dir):
                    shutil.rmtree(temp_outputs_dir)
            except Exception as cleanup_exception:
                logging.warning(f"Error during cleanup: {cleanup_exception}")

    def download_audio_and_convert_to_wav(self, file_url,temp_wav_filename):
        """Downloads an audio file from the given URL and converts it to a WAV file."""
        try:
            response = requests.get(file_url)
            response.raise_for_status()  # Check for HTTP errors
        except requests.RequestException as e:
            logging.error(f"Failed to download file from URL: {e}")
            raise
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_audio_file:
            temp_audio_filename = temp_audio_file.name
            temp_audio_file.write(response.content)

        command_ffmpeg = [
            'ffmpeg',
            '-i', temp_audio_filename,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            temp_wav_filename
        ]
        logging.info(f"Running FFmpeg command: {' '.join(command_ffmpeg)}")
        try:
            subprocess.run(
                command_ffmpeg,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            os.remove(temp_audio_filename)
            os.remove(temp_wav_filename)
            logging.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")

        os.remove(temp_audio_filename)
        return temp_wav_filename

    def separate_vocals(self, audio_path, output_dir):
        """Separates vocals from the audio using Demucs."""
        command_demucs = [
            'python3', '-m', 'demucs.separate',
            '-n', 'htdemucs',
            '--two-stems=vocals',
            audio_path,
            '-o', output_dir
        ]
        logging.info(f"Running Demucs command: {' '.join(command_demucs)}")
        try:
            subprocess.run(
                command_demucs,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Demucs separation failed: {e.stderr.decode()}")
            raise RuntimeError(f"Demucs separation failed: {e.stderr.decode()}")
