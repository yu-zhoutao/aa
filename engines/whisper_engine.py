import os
import io
import time
import json
import uuid
import wave
import math
import base64
import requests
import subprocess
import imageio_ffmpeg
import urllib3
from typing import List, Dict, Any, Tuple
from multiprocessing.pool import ThreadPool
from info_judge_next.config import Config

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WhisperEngine:
    """
    Online ASR Speech-to-Text Engine (Concurrent Version)
    Integrates audio transcoding, slicing, concurrent requests, and result merging.
    """

    @classmethod
    def _convert_to_16k_wav(cls, source_path: str) -> str:
        """
        Convert any audio/video to 16k mono WAV using FFmpeg
        """
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            os.makedirs(Config.FIXED_TEMP_DIR)

        filename = f"temp_asr_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(Config.FIXED_TEMP_DIR, filename)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe, '-y',
            '-i', source_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_path
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio transcoding failed: {e.stderr.decode() if e.stderr else 'unknown error'}")

    @classmethod
    def _split_wav(cls, byte_data: bytes, segment_length=60):
        """
        Slice WAV binary data by duration
        :param segment_length: Slice duration (seconds), default 60s
        """
        wf = wave.open(io.BytesIO(byte_data), "rb")
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()

        duration = nframes / framerate
        data_length = int(segment_length * framerate)

        segments = []
        num_chunks = math.ceil(1.0 * duration / segment_length)

        for i in range(num_chunks):
            wf.setpos(i * data_length)
            data = wf.readframes(data_length)

            tmpf = io.BytesIO()
            with wave.open(tmpf, "wb") as new_wf:
                new_wf.setnchannels(nchannels)
                new_wf.setsampwidth(sampwidth)
                new_wf.setframerate(framerate)
                new_wf.writeframes(data)

            segments.append(tmpf.getvalue())

        wf.close()
        return segments, duration

    @classmethod
    def _asr_infer(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single API request task
        """
        try:
            bdata = task["bdata"]
            url = Config.ASR_API_URL

            payload = {
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
                "audio_type": "wav",
                "audio_data": base64.b64encode(bdata).decode(),
                "stream": False,
                "audio_fs": 16000
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {Config.ASR_API_KEY}",
                "Connection": "keep-alive",
            }

            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                verify=False,
                proxies={"http": None, "https": None},
                timeout=60
            )

            if resp.status_code != 200:
                print(f"❌ ASR Chunk Failed: {resp.status_code} - {resp.text[:100]}")
                return {{}}

            jsres = resp.json()
            if "result" in jsres and "result" in jsres["result"]:
                result_list = jsres["result"]["result"]
                if result_list:
                    result = result_list[0]
                    # Fix relative timestamp to absolute timestamp
                    result["start"] = task["bg"] * 1000
                    return result

            return {{}}

        except Exception as e:
            print(f"❌ ASR Infer Exception: {e}")
            return {{}}

    @classmethod
    def _merge_asr_results(cls, results: List[Dict], punctuation="。！？；，、", ts_unit="ms"):
        """
        Merge results and segment sentences based on punctuation
        """
        full_text = ""
        timestamp_list = []  # [[start, end], ...]
        segments = []  # [{"start":, "end":, "text":}, ...]

        sorted_results = sorted([r for r in results if r], key=lambda x: x.get("start", 0))

        def to_sec(t):
            return float(t) / 1000.0 if ts_unit == "ms" else float(t)

        for r in sorted_results:
            full_text += r.get("text", "")
            chunk_start_ms = r.get("start", 0)
            for t in r.get("timestamp", []):
                t_start = to_sec(t[0] + chunk_start_ms)
                t_end = to_sec(t[1] + chunk_start_ms)
                timestamp_list.append([t_start, t_end])

        current_sentence = ""
        sent_start = None
        ts_idx = 0

        for char in full_text:
            if ts_idx < len(timestamp_list):
                if sent_start is None:
                    sent_start = timestamp_list[ts_idx][0]
                sent_end = timestamp_list[ts_idx][1]
                ts_idx += 1
            else:
                if sent_start is None: sent_start = 0.0
                sent_end = sent_start

            current_sentence += char

            if char in punctuation:
                clean_sentence = current_sentence.strip()
                if clean_sentence:
                    segments.append({
                        "start": round(sent_start, 2),
                        "end": round(sent_end, 2),
                        "text": clean_sentence
                    })
                current_sentence = ""
                sent_start = None

        if current_sentence.strip():
            segments.append({
                "start": round(sent_start if sent_start else 0.0, 2),
                "end": round(sent_end if sent_end else 0.0, 2),
                "text": current_sentence.strip()
            })

        return full_text, segments

    @classmethod
    def transcribe(cls, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main entry: Execute audio transcription
        :return: (Full Text, Detailed Segments)
        """
        wav_path = None
        try:
            wav_path = cls._convert_to_16k_wav(audio_path)

            with open(wav_path, "rb") as f:
                wav_bytes = f.read()

            segment_length = 60
            wav_chunks, duration = cls._split_wav(wav_bytes, segment_length)

            tasks = []
            for i, chunk_data in enumerate(wav_chunks):
                tasks.append({
                    "bg": i * segment_length,
                    "ed": min((i + 1) * segment_length, duration),
                    "bdata": chunk_data
                })

            # Use thread pool for concurrent requests (ASR_THREAD_POOL_SIZE not in Config? default to 6)
            pool_size = min(len(tasks), getattr(Config, 'ASR_THREAD_POOL_SIZE', 6))

            with ThreadPool(pool_size) as p:
                raw_results = p.map(cls._asr_infer, tasks)

            full_text, segments = cls._merge_asr_results(raw_results)

            return full_text, segments

        except Exception as e:
            print(f"❌ ASR Transcription Exception: {e}")
            return "", []

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except:
                    pass

    @classmethod
    def format_segments_for_llm(cls, segments: List[Dict[str, Any]]) -> str:
        formatted = []
        for s in segments:
            formatted.append(f"[{s['start']} - {s['end']}] {s['text']}")
        return "\n".join(formatted)
