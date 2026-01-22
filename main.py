import asyncio
import sys
import os

# 将 src 目录添加到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agent import JudgeAgent
from tools.visual_tools import (
    FrameExtractTool, 
    FrameUploadTool, 
    YoloDetectTool, 
    OcrDetectTool, 
    BehaviorJudgeTool
)

async def main():
    # 准备文件 (请确保这里有一个真实的测试文件，或者我会生成一个假的)
    # 为了演示，我们假设用户会提供一个路径
    file_path = "/Users/zhoutao/Desktop/360/DeepSearchAgent-Demo/img/1.png" # 使用已有的图片进行测试
    file_type = "image"
    
    if not os.path.exists(file_path):
        print(f"❌ 测试文件不存在: {file_path}")
        # 创建一个假的测试文件用于流程跑通
        with open("test_dummy.txt", "w") as f: f.write("dummy content")
        file_path = "test_dummy.txt"
        file_type = "unknown"

    print(f"🔍 开始审核文件: {file_path}")

    # 初始化工具箱
    tools = [
        FrameExtractTool(),
        FrameUploadTool(),
        YoloDetectTool(),
        OcrDetectTool(),
        BehaviorJudgeTool()
    ]
    
    # 初始化 Agent
    agent = JudgeAgent(tools=tools)
    
    # 执行审核
    try:
        state = await agent.audit(file_path, file_type)
        
        # 输出结果
        print("\n" + "="*50)
        print("📋 最终审核报告:")
        print("="*50)
        print(state.final_report)
        print("="*50)
        print(f"⚖️  违规判定: {"["违规"]" if state.is_violation else "["合规"]"}")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())