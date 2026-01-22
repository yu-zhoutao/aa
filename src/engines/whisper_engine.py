import os
import io
import time
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
from ..config import Config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class WhisperEngine:
    """
    在线 ASR 语音转写引擎 (并发版)
    """

    @classmethod
    def _convert_to_16k_wav(cls, source_path: str) -> str:
        config = Config()
        if not os.path.exists(config.temp_dir):
            os.makedirs(config.temp_dir)

        filename = f"temp_asr_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(config.temp_dir, filename)
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
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"音频转码失败: {e.stderr.decode() if e.stderr else 'unknown error'}")

    @classmethod
    def _split_wav(cls, byte_data: bytes, segment_length=60):
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
        config = Config()
        try:
            bdata = task["bdata"]
            url = config.asr_api_url
            if not url: return {}

            payload = {
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
                "audio_type": "wav",
                "audio_data": base64.b64encode(bdata).decode(),
                "stream": False,
                "audio_fs": 16000
            }

            headers = {
                "Content-Type": "application/json",
                "Connection": "keep-alive",
            }
            # 如果有 key 则添加
            if config.asr_api_key and config.asr_api_key != "EMPTY":
                headers["Authorization"] = f"Bearer {config.asr_api_key}"

            resp = requests.post(
                url, headers=headers, json=payload,
                verify=False, proxies={"http": None, "https": None}, timeout=60
            )

            if resp.status_code != 200:
                print(f"❌ ASR Chunk Failed: {resp.status_code}")
                return {}

            jsres = resp.json()
            if "result" in jsres and "result" in jsres["result"]:
                result_list = jsres["result"]["result"]
                if result_list:
                    result = result_list[0]
                    result["start"] = task["bg"] * 1000
                    return result

            return {}

        except Exception as e:
            print(f"❌ ASR Infer Exception: {e}")
            return {}

    @classmethod
    def _merge_asr_results(cls, results: List[Dict], punctuation="。！？；，、", ts_unit="ms"):
        full_text = ""
        timestamp_list = []
        segments = []

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
                if sent_start is None: sent_start = timestamp_list[ts_idx][0]
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
        config = Config()
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

            pool_size = min(len(tasks), config.asr_thread_pool_size)
            with ThreadPool(pool_size) as p:
                raw_results = p.map(cls._asr_infer, tasks)

            full_text, segments = cls._merge_asr_results(raw_results)
            return full_text, segments

        except Exception as e:
            print(f"❌ ASR 转写异常: {e}")
            return "", []

        finally:
            if wav_path and os.path.exists(wav_path):
                try: os.remove(wav_path)
                except: pass

    @classmethod
    def format_segments_for_llm(cls, segments: List[Dict[str, Any]]) -> str:
        formatted = []
        for s in segments:
            formatted.append(f"[{s['start']} - {s['end']}] {s['text']}")
        return "\n".join(formatted)