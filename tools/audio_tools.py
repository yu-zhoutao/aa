import os
import uuid
import asyncio
import time
import imageio_ffmpeg as ffmpeg
from typing import Dict, List, Any
from info_judge_next.config import Config
from info_judge_next.tools.base import BaseTool
from info_judge_next.engines.whisper_engine import WhisperEngine
from info_judge_next.engines.llm_client import LLMClient
from info_judge_next.agent.templates import PromptTemplates
from info_judge_next.utils.json_utils import JSONUtils

class AudioTranscribeTool(BaseTool):
    name = "audio_transcribe"
    description = "Audio transcription and violation detection tool. Extracts speech-to-text and detects political/vulgar/violent content. If violations are found, it slices the audio/video for evidence."

    async def run(self, file_path: str) -> Dict[str, Any]:
        """
        Executes audio transcription and violation detection.
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        # 1. Transcription (Whisper)
        loop = asyncio.get_running_loop()
        whisper_start_time = time.perf_counter()
        try:
            raw_text, segments = await loop.run_in_executor(
                None, WhisperEngine.transcribe, file_path
            )
            whisper_elapsed_time = time.perf_counter() - whisper_start_time
            print(f"⏱️ Whisper Transcription Time: {whisper_elapsed_time:.2f} s")
        except Exception as e:
            print(f"❌ Whisper Failed: {e}")
            return {"error": f"Transcription failed: {e}"}

        if not raw_text.strip():
            return {"status": "success", "text_content": "No valid speech", "violation_segments": []}

        # 2. Text Correction
        correction_prompt = PromptTemplates.audio_correction_prompt(raw_text)
        corrected_text = raw_text
        try:
            client = LLMClient.get_async_client()
            resp = await client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0.3
            )
            if resp.choices[0].message.content:
                corrected_text = resp.choices[0].message.content
        except Exception as e:
            print(f"Text correction failed, using original: {e}")

        # 3. Violation Check (LLM)
        formatted_text = WhisperEngine.format_segments_for_llm(segments)
        judge_prompt = PromptTemplates.text_review_and_correct_json_template(formatted_text)
        
        violation_report = {"is_violation": False, "segments": []}

        try:
            violation_data = await LLMClient.get_json_response([
                {"role": "user", "content": judge_prompt}
            ])
            
            if violation_data and violation_data.get("is_violation"):
                violation_report["is_violation"] = True
                merged_anchors = JSONUtils.merge_intervals(violation_data.get("time_anchors", []))
                
                # 4. Evidence Slicing
                clip_tasks = []
                for anchor in merged_anchors:
                    clip_tasks.append(self._slice_video(
                        file_path, anchor['start'], anchor['end']
                    ))
                
                clip_filenames = await asyncio.gather(*clip_tasks)
                
                for i, fname in enumerate(clip_filenames):
                    if fname:
                        # Construct URL accessible by frontend
                        merged_anchors[i]["clip_url"] = f"/static_temp/{fname}"
                
                violation_report["segments"] = merged_anchors
        
        except Exception as e:
            print(f"Audio violation check failed: {e}")

        return {
            "status": "success",
            "text_content": raw_text,
            "corrected_text": corrected_text,
            "violation_check": violation_report,
        }

    async def _slice_video(self, input_path: str, start: float, end: float) -> str:
        """Call ffmpeg to slice media file"""
        try:
            if not os.path.exists(Config.FIXED_TEMP_DIR):
                os.makedirs(Config.FIXED_TEMP_DIR)
            
            input_ext = os.path.splitext(input_path)[1].lower()
            audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
            is_audio_mode = input_ext in audio_extensions

            if is_audio_mode:
                output_ext = ".mp3"
                encoding_args = ['-vn', '-c:a', 'libmp3lame', '-q:a', '2']
            else:
                output_ext = ".mp4"
                encoding_args = ['-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-strict', 'experimental']
            
            output_filename = f"evidence_{uuid.uuid4().hex[:8]}{output_ext}"
            output_path = os.path.join(Config.FIXED_TEMP_DIR, output_filename)
            
            duration = max(end - start, 1.0)
            
            cmd = [
                ffmpeg.get_ffmpeg_exe(), '-y',
                '-ss', str(start), 
                '-t', str(duration),
                '-i', input_path,
            ] + encoding_args + [output_path]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return output_filename
            else:
                print(f"FFmpeg Error: {stderr.decode()}") 
                return ""
        except Exception as e:
            print(f"❌ Slice Exception: {e}")
            return ""

    def _get_args_schema(self) -> Dict:
        return {"file_path": {"type": "string", "description": "Local path to the media file"}}

    def _get_required_args(self) -> List[str]:
        return ["file_path"]
