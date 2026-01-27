import os
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from info_judge_next.config import Config
from info_judge_next.utils.file_utils import FileUtils
from info_judge_next.utils.sse_utils import SSEUtils
from info_judge_next.agent.core import AuditAgent

# Import Tools
from info_judge_next.tools.file_tools import MinioUploadTool, FrameExtractionTool
from info_judge_next.tools.vision_tools import OcrTool, FaceDetectionTool, YoloDetectionTool
from info_judge_next.tools.annotation_tool import ImageAnnotationTool
from info_judge_next.tools.search_tools import WebSearchTool
from info_judge_next.tools.audio_tools import AudioTranscribeTool  # Added

app = FastAPI(title="重构版审核智能体", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(Config.FIXED_TEMP_DIR):
    os.makedirs(Config.FIXED_TEMP_DIR)
app.mount("/static_temp", StaticFiles(directory=Config.FIXED_TEMP_DIR), name="static_temp")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "ReAct Agent", "timestamp": time.time()}

@app.post("/analyze")
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        file_path = FileUtils.save_upload_file(file)
        file_type = FileUtils.detect_file_type(file.filename)
    except Exception as e:
        async def error_handler():
            yield SSEUtils.error(f"文件上传错误: {str(e)}")
        return StreamingResponse(error_handler(), media_type="text/event-stream")

    # Initialize Toolkit
    tools = [
        MinioUploadTool(),
        FrameExtractionTool(),
        OcrTool(),
        FaceDetectionTool(),
        YoloDetectionTool(),
        ImageAnnotationTool(),
        WebSearchTool(),
        AudioTranscribeTool() # Registered
    ]

    agent = AuditAgent(tools=tools)

    async def stream_factory():
        try:
            async for event in agent.execute(file_path, file_type):
                yield event
        except Exception as e:
            yield SSEUtils.error(f"智能体错误: {str(e)}")
            
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)