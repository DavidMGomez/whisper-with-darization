import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
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
import confirmation
import nemotron_diarization
from transcription_helpers import transcribe_batched
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
import gc

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



compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
WHISPER_MODEL_PATHS = {
    "large-v3": "./models/faster-whisper-large-v3",
    "large-v3-turbo": "./models/faster-whisper-large-v3-turbo",
}

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



def get_sentences_speaker_mapping( sentences, audio):
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
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'
        os.makedirs(destination_folder, exist_ok=True)
        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)
            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

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
        whisper_model: str = Input(
            description="Which faster-whisper model to transcribe with. large-v3-turbo is "
                        "2-4x faster with a small (~1-2%) WER increase; large-v3 is kept for "
                        "rollback if turbo's accuracy isn't good enough for a given use case.",
            default="large-v3-turbo",
            choices=["large-v3", "large-v3-turbo"]
        ),
        multimedia_part_id: str = Input(
            description="Multimedia part ID", default=None
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
            description="Deprecated, unused since diarization no longer depends on a gated "
                        "HuggingFace model. Kept only so existing callers that pass it don't break.",
            default=None
        ),
        min_num_speakers: int = Input(
            description="Min number of speakers", default=None
        ),
        max_num_speakers: int = Input(
            description="Max number of speakers", default=None
        ),
        use_speaker_confirmation: bool = Input(
            description="If true, ambiguous speaker-turn assignments (low diarization confidence "
                        "or a large time gap since the previous turn) are double-checked against "
                        "conversational context using confirmation_provider.",
            default=False
        ),
        confirmation_provider: str = Input(
            description="Which confirmation provider to use when use_speaker_confirmation is true. "
                        "See confirmation.PROVIDERS for the registry of supported providers.",
            default="jev"
        ),
        jev_api_key: str = Input(
            description="API key for the 'jev' confirmation provider (ignored for other "
                        "providers). Falls back to the JEV_API_KEY env var if not set.",
            default=None
        ),
        confirmation_confidence_threshold: float = Input(
            description="Diarization segments with a speaker-confidence score below this "
                        "(0-1) are sent for confirmation when use_speaker_confirmation is true.",
            default=0.55
        ),
        confirmation_gap_threshold_seconds: float = Input(
            description="Segments preceded by a silence gap longer than this (seconds) are sent "
                        "for confirmation when use_speaker_confirmation is true, since a long "
                        "pause makes a speaker change more likely.",
            default=1.5
        )
    ) -> Output:
        if file_url is None:
            raise ValueError("ERROR: 'file_url' is required!")

        random_uuid = uuid.uuid4()
        vocal_target  = f"temp-{random_uuid}.wav"

        try:
            # Download and convert audio to WAV
            vocal_target = self.download_audio_and_convert_to_wav(file_url,vocal_target)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            start_time = time.time_ns() / 1e6
            
            whisper_arch = WHISPER_MODEL_PATHS[whisper_model]
            model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, language=language,
                                        asr_options={"temperatures": [0]}, vad_options={"vad_onset": 0.500,"vad_offset": 0.363})
            
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(vocal_target)

            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to load audio: {elapsed_time:.2f} ms")

          
            start_time = time.time_ns() / 1e6
            
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]
            print(f"language: {detected_language}")
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if language in wav2vec2_langs:
                result = self.align(audio, result)
                result, segment_confidences = self.diarize(audio, result, max_num_speakers)
                # Get sentences with speaker mapping
                segments = get_sentences_speaker_mapping(
                    result["segments"],
                    AudioSegment.from_file(vocal_target).set_channels(1)
                )

                if use_speaker_confirmation:
                    # jev_api_key only applies to the "jev" provider; a future provider would get
                    # its own <provider>_api_key input wired in here the same way.
                    provider_api_key = jev_api_key if confirmation_provider == "jev" else None
                    provider = confirmation.get_confirmation_provider(confirmation_provider, provider_api_key)
                    segments = confirmation.annotate_and_confirm(
                        segments,
                        segment_confidences,
                        confidence_threshold=confirmation_confidence_threshold,
                        gap_threshold_seconds=confirmation_gap_threshold_seconds,
                        provider=provider,
                    )

                # Send success message to Pub/Sub if credentials are provided
                if credentials and project_id and topic_id and multimedia_part_id:
                    send_pubsub_message(
                        project_id,
                        topic_id,
                        {"id": multimedia_part_id, "status": "success"},
                        credentials
                    )

                return Output(segments=segments)

            else:
                # Handle case where language is not supported
                raise ValueError(f"Language '{language}' is not supported for alignment.")

        except Exception as e:
            # Send failure message to Pub/Sub if credentials are provided
            if credentials and project_id and topic_id and multimedia_part_id:
                send_pubsub_message(
                    project_id,
                    topic_id,
                    {"id": multimedia_part_id, "status": "failed", "error": str(e)},
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
            except Exception as cleanup_exception:
                logging.warning(f"Error during cleanup: {cleanup_exception}")

    def diarize(self, audio, result, max_speakers):
        start_time = time.time_ns() / 1e6

        max_num_speakers = max_speakers if max_speakers else 8
        diarize_segments, segment_confidences = nemotron_diarization.diarize(
            audio, max_num_speakers=max_num_speakers
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

        gc.collect()
        torch.cuda.empty_cache()
        nemotron_diarization.unload()

        return result, segment_confidences

    def align(self, audio, result):
        start_time = time.time_ns() / 1e6

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device,return_char_alignments=False)
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")
        gc.collect()
        torch.cuda.empty_cache()
        del model_a

        return result
    
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
