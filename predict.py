# Prediction interface for Cog ⚙️
from typing import Any, List
import numpy as np
import subprocess
import os
import pandas as pd
import requests
import time
import torch
import re
import whisperx
import uuid
from helpers import *
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from cog import BasePredictor, BaseModel, Input, File, Path
import numpy as np
from random import choice

mtypes = {"cpu": "int8", "cuda": "float16"}

class Output(BaseModel):
    segments: list


class Predictor(BasePredictor):
    def setup(self):
        pass
        
    def predict(
        self,
        file_url: str = Input(
            description="Or provide: A direct audio file URL", default=None),
        language: str = Input(description="Language spoken in the audio, specify None to perform language detection",
                            default="es"),
        batch_size: int = Input(description="Batch size for batched inference",
                            default=8)
    ) -> Output:
        random_uuid = uuid.uuid4()
        vocal_target  = f"temp-{random_uuid}.wav"
        folder_outs = f"temp_{random_uuid}_outputs"
    
        try:
            if file_url is not None:
                self.download_audio_and_convert_to_wav(file_url,vocal_target)
                command_demucs = f'python -m demucs.separate -n htdemucs --two-stems=vocals "./{vocal_target}" -o "{folder_outs}"'
                print("Separando voces ")
                print(command_demucs)
                return_code = os.system(command_demucs)
                print(os.listdir("./"+folder_outs))
                if return_code == 0:
                    vocal_target = os.path.join(
                        folder_outs,
                        "htdemucs",
                        os.path.splitext(os.path.basename(vocal_target))[0],
                        "vocals.wav",
                    )
                print(vocal_target)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                from transcription_helpers import transcribe_batched
                whisper_results, language = transcribe_batched(
                    vocal_target,
                    language,
                    batch_size,
                    "large-v2",
                    mtypes[device],
                    False,
                    device,
                )
                print("search "+str(language)+" in "+str(wav2vec2_langs))
                if language in wav2vec2_langs:
                    alignment_model, metadata = whisperx.load_align_model(
                        language_code=language, device=device
                    )
                    result_aligned = whisperx.align(
                        whisper_results, alignment_model, metadata, vocal_target, device
                    )
                    word_timestamps = filter_missing_timestamps(
                        result_aligned["word_segments"],
                        initial_timestamp=whisper_results[0].get("start"),
                        final_timestamp=whisper_results[-1].get("end"),
                    )
                    # clear gpu vram
                    del alignment_model
                    torch.cuda.empty_cache()
                else:
                    assert (
                        batch_size == 0  # TODO: add a better check for word timestamps existence
                    ), (
                        f"Unsupported language: {language}, use --batch_size to 0"
                        " to generate word timestamps using whisper directly and fix this error."
                    )
                    word_timestamps = []
                    for segment in whisper_results:
                        for word in segment["words"]:
                            word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})
                
                print(word_timestamps)
                # convert audio to mono for NeMo combatibility
                sound = AudioSegment.from_file(vocal_target).set_channels(1)
                ROOT = os.getcwd()
                temp_path = os.path.join(ROOT, "temp_outputs")
                os.makedirs(temp_path, exist_ok=True)
                sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")
                # Initialize NeMo MSDD diarization model
                msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
                msdd_model.diarize()
                del msdd_model
                torch.cuda.empty_cache()
                speaker_ts = []
                with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line_list = line.split(" ")
                        s = int(float(line_list[5]) * 1000)
                        e = s + int(float(line_list[8]) * 1000)
                        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
                wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
                if language in punct_model_langs:
                    # restoring punctuation in the transcript to help realign the sentences
                    punct_model = PunctuationModel(model="kredor/punctuate-all")

                    words_list = list(map(lambda x: x["word"], wsm))

                    labled_words = punct_model.predict(words_list)

                    ending_puncts = ".?!"
                    model_puncts = ".,;:!?"

                    # We don't want to punctuate U.S.A. with a period. Right?
                    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

                    for word_dict, labeled_tuple in zip(wsm, labled_words):
                        word = word_dict["word"]
                        if (
                            word
                            and labeled_tuple[1] in ending_puncts
                            and (word[-1] not in model_puncts or is_acronym(word))
                        ):
                            word += labeled_tuple[1]
                            if word.endswith(".."):
                                word = word.rstrip(".")
                            word_dict["word"] = word
                else:
                    logging.warning(
                            f"Punctuation restoration is not available for {language} language. Using the original punctuation.")
                   
                wsm = get_realigned_ws_mapping_with_punctuation(wsm)
                ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
                return Output(segments=ssm)

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)
                
        finally:
            # Clean up
            if os.path.exists(vocal_target):
                os.remove(vocal_target)
    
    def download_audio_and_convert_to_wav(self, file_url, temp_wav_filename):
        response = requests.get(file_url)
        temp_audio_filename = f"temp-{time.time_ns()}.mp4"
        with open(temp_audio_filename, 'wb') as file:
            file.write(response.content)
        command_ffmpeg = f'ffmpeg -i {temp_audio_filename} -ar 16000 -ac 1 -c:a pcm_s16le {temp_wav_filename}'
        print(command_ffmpeg)
        return_code = os.system(command_ffmpeg)
        print(return_code)
        if os.path.exists(temp_audio_filename):
            print(os.listdir("./"))
            print("removing "+temp_audio_filename)
            os.remove(temp_audio_filename)
