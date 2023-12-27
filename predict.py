# Prediction interface for Cog ⚙️
from typing import Any, List
import base64
import contextlib
import datetime
import json
import magic
import mimetypes
import numpy as np
import subprocess
import io
import os
import pandas as pd
import requests
import time
import torch
import wave
import re

from cog import BasePredictor, BaseModel, Input, File, Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio import Model
from pyannote.audio import Inference
import random
import numpy as np
import librosa 
from deep_speaker.audio import read_mfcc,mfcc_fbank
from deep_speaker.batcher import pad_mfcc
from random import choice
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity


def sample_from_mfcc(mfcc, max_length):
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r:r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)




class Output(BaseModel):
    segments: list


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "large-v2"
        self.model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16")
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token="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").to(
                torch.device("cuda"))
        # Define the model here.
        
        
        
    def sample_from_mfcc(self,mfcc, max_length):
        if mfcc.shape[0] >= max_length:
            r = choice(range(0, len(mfcc) - max_length + 1))
            s = mfcc[r:r + max_length]
        else:
            s = pad_mfcc(mfcc, max_length)
        return np.expand_dims(s, axis=-1)
    
    def read_segment_mfcc(self, path,segment, sample_rate):
        audio = self.read_audio_segment_mfcc(segment,path,SAMPLE_RATE)
        energy = np.abs(audio)
        silence_threshold = np.percentile(energy, 95)
        offsets = np.where(energy > silence_threshold)[0]
        audio_voice_only = audio[offsets[0]:offsets[-1]]
        mfcc = mfcc_fbank(audio_voice_only, sample_rate)
        return mfcc

    
    def read_audio_segment_mfcc(self, segment, path,sample_rate):
        # Convertir start y end a flotantes
        start = float(segment["start"])
        end = float(segment["end"])
        segment_start = start
        segment_end = end
        y, sr = librosa.load(path, sr=sample_rate, offset=segment_start, duration=segment_end - segment_start, mono=True, dtype=np.float32)
        return y


    def segment_embedding(self,
                          segment,
                          path):
        mfcc = self.sample_from_mfcc(self.read_segment_mfcc(path, segment ,SAMPLE_RATE),NUM_FRAMES)
        return self.deep_speaker_model.m.predict(np.expand_dims(mfcc, axis=0))

    def predict(
        self,
        file_string: str = Input(
            description="Either provide: Base64 encoded audio file,",
            default=None),
        file_url: str = Input(
            description="Or provide: A direct audio file URL", default=None),
        file: Path = Input(description="Or an audio file", default=None),
        group_segments: bool = Input(
            description=
            "Group segments of same speaker shorter apart than 2 seconds",
            default=True),
        num_speakers: int = Input(description="Number of speakers",
                                  ge=1,
                                  le=50,
                                  default=2),
        prompt: str = Input(description="Prompt, to be used as context",
                            default="Some people speaking."),
        offset_seconds: int = Input(
            description="Offset in seconds, used for chunked inputs",
            default=0,
            ge=0)
    ) -> Output:
        """Run a single prediction on the model"""
        # Check if either filestring, filepath or file is provided, but only 1 of them
        """ if sum([file_string is not None, file_url is not None, file is not None]) != 1:
            raise RuntimeError("Provide either file_string, file or file_url") """

        try:
            # Generate a temporary filename
            temp_wav_filename = f"temp-{time.time_ns()}.wav"

            if file is not None:
                subprocess.run([
                    'ffmpeg', '-i', file, '-ar', '16000', '-ac', '1', '-c:a',
                    'pcm_s16le', temp_wav_filename
                ])

            elif file_url is not None:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, 'wb') as file:
                    file.write(response.content)

                subprocess.run([
                    'ffmpeg', '-i', temp_audio_filename, '-ar', '16000', '-ac',
                    '1', '-c:a', 'pcm_s16le', temp_wav_filename
                ])

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)
            elif file_string is not None:
                audio_data = base64.b64decode(
                    file_string.split(',')[1] if ',' in
                    file_string else file_string)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, 'wb') as f:
                    f.write(audio_data)

                subprocess.run([
                    'ffmpeg', '-i', temp_audio_filename, '-ar', '16000', '-ac',
                    '1', '-c:a', 'pcm_s16le', temp_wav_filename
                ])

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)

            segments = self.speech_to_text(temp_wav_filename, num_speakers,
                                           prompt, offset_seconds,
                                           group_segments)

            print(f'done with inference')
            # Return the results as a JSON object
            return Output(segments=segments)

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(self,
                       audio_file_wav,
                       num_speakers=2,
                       prompt="People takling.",
                       offset_seconds=0,
                       group_segments=True):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        options = dict(vad_filter=True,
                       initial_prompt=prompt,
                       word_timestamps=True)
        segments, _ = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [{
            'start':
            float(s.start + offset_seconds),
            'end':
            float(s.end + offset_seconds),
            'text':
            s.text,
            'words': [{
                'start': float(w.start + offset_seconds),
                'end': float(w.end + offset_seconds),
                'word': w.word
            } for w in s.words]
        } for s in segments]

        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds"
        )
        diarization = self.diarization_model(audio_file_wav,
                                             num_speakers=num_speakers)

        time_diraization_end = time.time()
        print(
            f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds"
        )

        # Initialize variables to keep track of the current position in both lists
        margin = 0.1  # 0.1 seconds margin

        # Initialize an empty list to hold the final segments with speaker info
        final_segments = []

        diarization_list = list(diarization.itertracks(yield_label=True))
        speaker_idx = 0
        n_speakers = len(diarization_list)

        # Iterate over each segment
        for segment in segments:
            segment_start = segment['start'] + offset_seconds
            segment_end = segment['end'] + offset_seconds
            segment_text = []
            segment_words = []

            # Iterate over each word in the segment
            for word in segment['words']:
                word_start = word['start'] + offset_seconds - margin
                word_end = word['end'] + offset_seconds + margin

                while speaker_idx < n_speakers:
                    turn, _, speaker = diarization_list[speaker_idx]

                    if turn.start <= word_end and turn.end >= word_start:
                        # Add word without modifications
                        segment_text.append(word['word'])
                        
                        # Strip here for individual word storage
                        word['word'] = word['word'].strip()
                        segment_words.append(word)

                        if turn.end <= word_end:
                            speaker_idx += 1

                        break
                    elif turn.end < word_start:
                        speaker_idx += 1
                    else:
                        break

            if segment_text:
                combined_text = ''.join(segment_text)
                cleaned_text = re.sub('  ', ' ', combined_text).strip()
                new_segment = {
                    'start': segment_start - offset_seconds,
                    'end': segment_end - offset_seconds,
                    'speaker': speaker,
                    'text': cleaned_text,
                    'words': segment_words
                }
                final_segments.append(new_segment)

        time_merging_end = time.time()
        print(
            f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds"
        )
        segments = final_segments
        # Make output
        output = []  # Initialize an empty list for the output

        self.deep_speaker_model = DeepSpeakerModel()      
        self.deep_speaker_model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)
        
        for i in range(0, len(segments)):
            current_group = {
                'start': str(segments[i]["start"]),
                'end': str(segments[i]["end"]),
                'speaker': segments[i]["speaker"],
                'text': segments[i]["text"],
                'words': segments[i]["words"],
                'embeddingSpeaker':[]
            }
            embedding_speaker = self.segment_embedding(current_group,audio_file_wav)
            current_group["embeddingSpeaker"] = embedding_speaker
            output.append(current_group)

        time_cleaning_end = time.time()
        print(
            f"Finished with cleaning, took {time_cleaning_end - time_merging_end:.5} seconds"
        )
        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""Processing time: {time_diff:.5} seconds"""
        print(system_info)
        return output