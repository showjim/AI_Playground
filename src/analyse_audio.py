import os
import ffmpeg
import subprocess
from faster_whisper import WhisperModel
from tqdm import tqdm
import time as tt
import sys
from pathlib import Path

# for speaker identify
import datetime
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=device)

from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

is_vad_filter = "False"
# file_type = "audio"  # @param ["audio","video"]
# model_size = "base"  # @param ["base","small","medium", "large-v1","large-v2"]
# language = "zh"  # @param {type:"string"}
export_srt = "Yes"  # @param ["No","Yes"]
# num_speakers = 3 #@param {type:"integer"}

def extract_subtitle(file_names:str, file_type, language, model_size):
    print('语音识别库配置完毕，将开始转换')
    print('加载模型 Loading model...')
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    model = WhisperModel(model_size) #, device=device_str)

    # for i in range(len(file_names)):
    file_name = file_names#[i]
    # Transcribe
    file_basename = Path(file_name).stem
    file_dir = Path(file_name).parent
    output_file = str(Path(file_dir)/file_basename)
    if file_type == "video":
        print('提取音频中 Extracting audio from video file...')
        # os.system(f'ffmpeg -i {file_name} -f mp3 -ab 192000 -vn {file_basename}.mp3')
        os.system(f'ffmpeg -i {file_name} {output_file}.wav -y')
        print('提取完毕 Done.')
        file_name = output_file + ".wav"
        # print(file_basename)
    tic = tt.time()
    print('识别中 Transcribe in progress...')
    segments, info = model.transcribe(audio=f'{file_name}',
                                      beam_size=5,
                                      language=language,
                                      vad_filter=is_vad_filter,
                                      initial_prompt="Hello, welcome to my lecture.", # to help recognise punctuation
                                      vad_parameters=dict(min_silence_duration_ms=1000))

    # segments is a generator so the transcription only starts when you iterate over it
    # to use pysubs2, the argument must be a segment list-of-dicts
    total_duration = round(info.duration, 2)  # Same precision as the Whisper timestamps.
    results = []
    results_txt = []
    with tqdm(total=total_duration, unit=" seconds") as pbar:
        for s in segments:
            segment_dict = {'start': s.start, 'end': s.end, 'text': s.text}
            results.append(segment_dict)
            # store as txt
            results_txt.append(s.text)

            segment_duration = s.end - s.start
            pbar.update(segment_duration)

    # Time comsumed
    toc = tt.time()
    print('识别完毕 Done')
    print(f'Time consumpution {toc - tic}s')

    srt_string = ""
    if export_srt == "Yes":
        import pysubs2
        subs = pysubs2.load_from_whisper(results)
        # subs.save(output_file + '.srt')
        srt_string = subs.to_string("srt")

        with open(output_file + '.txt', 'w') as fp:
            for item in results_txt:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')
    return results, file_name, srt_string #str(output_file) + ".wav"

def identify_speaker(file_name, segments, num_speakers):
    print('Embedding audio to tensor...')
    embeddings = embedding_audio(file_name, segments)
    print('Identify the speakers...')
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    return segments

def embedding_audio(path, segments):
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(path, segment, audio, duration)
    embeddings = np.nan_to_num(embeddings)
    return embeddings

def segment_embedding(path, segment, audio, duration):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    tmp = waveform[None][:,0,:][None]
    return embedding_model(tmp)

def time(secs):
  return datetime.timedelta(seconds=round(secs))

def output_subtitle(path, segments):
    file_basename = Path(path).stem
    parent_dir = str(Path(path).parent)
    with open(parent_dir + "/output/"+file_basename + ".txt", "w", encoding="utf-8") as f:
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            f.write(segment["text"][1:] + ' ')


if __name__ == '__main__':
    path = "/Users/jerryzhou/PycharmProjects/AutoMeetingMinute/test2.mp4"
    segments, new_file = extract_subtitle(path)
    # embeddings = embedding_audio(new_file, segments)
    segments_speaker = identify_speaker(new_file, segments)
    output_subtitle(new_file, segments_speaker)
