import os
import uvicorn
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.utils.config import Config
from src.utils.file_utils import FileUtils
from src.utils.sse_utils import SSEUtils
from src.agent import JudgeAgent
from src.tools.visual_tools import (
    FrameExtractTool,
    FrameUploadTool,
    PreviewUploadTool,
    FaceIdentifyTool,
    YoloDetectTool,
    OcrDetectTool,
    BehaviorJudgeTool
)
from src.tools.audio_tools import (
    AudioTranscribeTool,
    AudioCorrectTool,
    AudioViolationCheckTool,
    AudioSliceTool
)
from src.tools.search_tools import WebSearchTool

app = FastAPI(
    title="JudgeAgent API",
    description="Refactored DeepSearch-style Audit Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()
if not os.path.exists(config.temp_dir):
    os.makedirs(config.temp_dir)
app.mount("/static_temp", StaticFiles(directory=config.temp_dir), name="static_temp")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "JudgeAgent"}

@app.post("/analyze")
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_search: bool = Form(True),
    enable_cache: bool = Form(True)
):
    try:
        file_path, minio_url = FileUtils.save_upload_file(file)
        file_type = FileUtils.detect_file_type(file.filename)
    except Exception as e:
        async def error_handler():
            yield SSEUtils.error(f"文件接收失败: {str(e)}")
        return StreamingResponse(error_handler(), media_type="text/event-stream")

    # 初始化所有工具
    tools = [
        # 视觉
        FrameExtractTool(),
        FrameUploadTool(),
        PreviewUploadTool(), # 新增
        FaceIdentifyTool(),  # 新增
        YoloDetectTool(),
        OcrDetectTool(),
        BehaviorJudgeTool(),
        # 听觉
        AudioTranscribeTool(),
        AudioCorrectTool(),
        AudioViolationCheckTool(),
        AudioSliceTool(),
        # 搜索
        WebSearchTool()      # 新增
    ]

    agent = JudgeAgent(tools=tools)

    async def stream_factory():
        async for event in agent.audit(file_path, file_type):
            yield event

    background_tasks.add_task(FileUtils.clear_temp_dir, age_seconds=3600)

    return StreamingResponse(
        stream_factory(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
