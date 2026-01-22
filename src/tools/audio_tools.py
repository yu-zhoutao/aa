import os
import uuid
import asyncio
import time
import imageio_ffmpeg as ffmpeg
from typing import Dict, List, Any
from ..utils.config import Config
from .base import BaseTool
from ..engines.whisper_engine import WhisperEngine
from ..engines.llm_engine import LLMEngine
from ..utils.prompts import PromptTemplates
from ..utils.json_utils import JSONUtils

def _ensure_temp_dir() -> None:
    config = Config()
    if not os.path.exists(config.temp_dir):
        os.makedirs(config.temp_dir)

class AudioTranscribeTool(BaseTool):
    def __init__(self):
        super().__init__("audio_transcribe", "语音转写工具（Whisper）")

    async def run(self, file_path: str, **kwargs) -> Dict[str, Any]:
        if not file_path or not os.path.exists(file_path):
            return {"error": "文件不存在"}

        loop = asyncio.get_running_loop()
        whisper_start_time = time.perf_counter()
        try:
            raw_text, segments = await loop.run_in_executor(
                None, WhisperEngine.transcribe, file_path
            )
            whisper_elapsed_time = time.perf_counter() - whisper_start_time
            print(f"⏱️ Whisper 转录耗时: {whisper_elapsed_time:.2f} 秒")
        except Exception as e:
            print(f"❌ Whisper 转录失败: {e}")
            return {"error": f"转写失败: {e}"}

        if not raw_text.strip():
            return {"status": "success", "text_content": "无有效语音", "segments": []}

        return {"status": "success", "text_content": raw_text, "segments": segments}

class AudioCorrectTool(BaseTool):
    def __init__(self):
        super().__init__("audio_correct", "语音文本纠错")

    async def run(self, text_content: str, **kwargs) -> Dict[str, Any]:
        if not text_content:
            return {"error": "缺少 text_content"}

        correction_prompt = PromptTemplates.audio_correction_prompt(text_content)
        corrected_text = text_content
        try:
            client = LLMEngine.get_client() 
            corrected_text = await client.ainvoke(system_prompt="You are a text corrector.", user_prompt=correction_prompt)

        except Exception as e:
            print(f"文本纠错失败，使用原文: {e}")

        return {"status": "success", "corrected_text": corrected_text}

class AudioViolationCheckTool(BaseTool):
    def __init__(self):
        super().__init__("audio_violation_check", "语音违规检测与切片")

    async def run(self, segments: List[Dict[str, Any]], file_path: str = "", **kwargs) -> Dict[str, Any]:
        """
        :param file_path: 原始文件路径，用于切片
        """
        if not segments:
            return {"error": "缺少 segments"}

        formatted_text = WhisperEngine.format_segments_for_llm(segments)
        judge_prompt = PromptTemplates.text_review_and_correct_json_template(formatted_text)

        violation_report = {"is_violation": False, "segments": []}
        clips = []
        
        try:
            msgs = [{"role": "user", "content": judge_prompt}]
            violation_data = await LLMEngine.get_json_response(msgs)

            if violation_data and violation_data.get("is_violation"):
                violation_report["is_violation"] = True
                merged_anchors = JSONUtils.merge_intervals(violation_data.get("time_anchors", []))
                violation_report["segments"] = merged_anchors
                
                # 如果有原文件，自动执行切片
                if file_path and os.path.exists(file_path):
                    slicer = AudioSliceTool()
                    res = await slicer.run(file_path, merged_anchors)
                    clips = res.get("clips", [])
                    print(f"✂️ 已生成 {len(clips)} 个违规音频/视频切片证据")

        except Exception as e:
            print(f"音频合规性检测失败: {e}")

        return {
            "status": "success",
            "violation_check": violation_report,
            "evidence": {
                "audio_risk": violation_report["is_violation"],
                "clips": clips, # 返回切片路径
                "segments": violation_report["segments"]
            },
        }

class AudioSliceTool(BaseTool):
    def __init__(self):
        super().__init__("audio_slice", "音频/视频切片")

    async def run(self, file_path: str, time_anchors: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        if not file_path or not os.path.exists(file_path):
            return {"error": "文件不存在"}
        if not time_anchors:
            return {"error": "缺少 time_anchors"}

        _ensure_temp_dir()
        clip_tasks = [
            self._slice_video(file_path, anchor["start"], anchor["end"])
            for anchor in time_anchors
            if "start" in anchor and "end" in anchor
        ]
        clip_filenames = await asyncio.gather(*clip_tasks)
        config = Config()

        clips = []
        for i, fname in enumerate(clip_filenames):
            if not fname: continue
            clip_path = os.path.join(config.temp_dir, fname)
            clips.append({"index": i, "clip_path": clip_path, "file": fname})

        return {"status": "success", "clips": clips}

    async def _slice_video(self, input_path: str, start: float, end: float) -> str:
        try:
            _ensure_temp_dir()
            config = Config()
            ext = os.path.splitext(input_path)[1] or ".mp4"
            output_filename = f"evidence_{uuid.uuid4().hex[:8]}{ext}"
            output_path = os.path.join(config.temp_dir, output_filename)
            duration = max(end - start, 1.0)

            cmd = [
                ffmpeg.get_ffmpeg_exe(), "-y",
                "-ss", str(start),
                "-t", str(duration),
                "-i", input_path,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                "-strict", "experimental",
                output_path,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode == 0:
                return output_filename
            return ""
        except Exception as e:
            print(f"❌ 视频切片异常: {e}")
            return ""