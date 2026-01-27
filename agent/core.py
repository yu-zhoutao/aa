import json
import asyncio
from typing import List, AsyncGenerator, Dict, Any
from info_judge_next.config import Config
from info_judge_next.engines.llm_client import LLMClient
from info_judge_next.utils.sse_utils import SSEUtils
from info_judge_next.tools.base import BaseTool
from info_judge_next.agent.prompts import SYSTEM_PROMPT

class AgentMemory:
    def __init__(self):
        self.messages = []
        self._finished = False
    
    def add_message(self, role: str, content: str = None, tool_calls=None, tool_call_id: str = None):
        msg = {"role": role}
        if content is not None: msg["content"] = content
        if tool_calls is not None: msg["tool_calls"] = tool_calls
        if tool_call_id is not None: msg["tool_call_id"] = tool_call_id
        self.messages.append(msg)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages

class AuditAgent:
    def __init__(self, tools: List[BaseTool]):
        self.tools_map = {t.name: t for t in tools}
        self.tools_schemas = [t.to_schema() for t in tools]
        self.client = LLMClient.get_async_client()
        self.model_name = Config.MODEL_NAME

    async def execute(self, file_path: str, file_type: str) -> AsyncGenerator[str, None]:
        memory = AgentMemory()
        memory.add_message("system", SYSTEM_PROMPT)
        memory.add_message("user", f"请审核该文件。路径: {file_path}, 类型: {file_type}")

        yield SSEUtils.log(f"🤖 智能体启动。加载工具数: {len(self.tools_map)}")
        print(f"\n[Agent Init] Loading {len(self.tools_map)} tools: {list(self.tools_map.keys())}")

        max_steps = 15
        step_count = 0

        while step_count < max_steps:
            step_count += 1
            yield SSEUtils.log(f"🤔 第 {step_count} 步思考中...")
            print(f"\n--- [Step {step_count}] Thinking ---")

            try:
                # 1. LLM 决策
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=memory.get_messages(),
                    tools=self.tools_schemas,
                    tool_choice="auto", 
                    temperature=0.1,
                )
                
                ai_message = response.choices[0].message
                memory.add_message(
                    role="assistant", 
                    content=ai_message.content, 
                    tool_calls=ai_message.tool_calls
                )

                # 2. 处理工具调用
                if ai_message.tool_calls:
                    if ai_message.content:
                        yield SSEUtils.token(f"\n> **思考**: {ai_message.content}\n\n")
                        print(f"[Thought]: {ai_message.content}")
                    
                    yield SSEUtils.log(f"⚡️ 正在执行 {len(ai_message.tool_calls)} 个工具任务...")
                    print(f"[Action] Triggering {len(ai_message.tool_calls)} tool calls.")
                    
                    # 并行执行
                    tasks = []
                    tool_meta = []

                    for tool_call in ai_message.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args_str = tool_call.function.arguments
                        
                        try:
                            fn_args = json.loads(fn_args_str)
                        except:
                            yield SSEUtils.error(f"❌ 参数解析错误: {fn_args_str}")
                            print(f"❌ [Error] Failed to parse args for {fn_name}: {fn_args_str}")
                            continue

                        yield SSEUtils.log(f"🚀 [调用] {fn_name}")
                        print(f"  -> Calling {fn_name} with args: {str(fn_args)[:200]}...") # Truncate long args
                        
                        if fn_name in self.tools_map:
                            tool_instance = self.tools_map[fn_name]
                            tasks.append(tool_instance.run(**fn_args))
                            tool_meta.append({"call": tool_call, "name": fn_name})
                        else:
                            tasks.append(None)
                            tool_meta.append({"call": tool_call, "name": fn_name, "error": "工具未找到"})

                    if tasks:
                        results = await asyncio.gather(*[t for t in tasks if t is not None], return_exceptions=True)
                        
                        res_iter = iter(results)
                        for meta in tool_meta:
                            if "error" in meta:
                                memory.add_message("tool", content=json.dumps({"error": meta["error"]}), tool_call_id=meta["call"].id)
                                print(f"  <- {meta['name']} failed: {meta['error']}")
                                continue
                            
                            res = next(res_iter)
                            if isinstance(res, Exception):
                                error_msg = f"工具执行错误: {str(res)}"
                                yield SSEUtils.error(f"❌ {meta['name']}: {error_msg}")
                                memory.add_message("tool", content=json.dumps({"error": error_msg}), tool_call_id=meta["call"].id)
                                print(f"  <- {meta['name']} exception: {error_msg}")
                            else:
                                # 成功
                                result_dict = res
                                yield SSEUtils.log(f"✅ [完成] {meta['name']}")
                                
                                # Log result summary to console
                                res_str = str(result_dict)
                                print(f"  <- {meta['name']} finished. Result size: {len(res_str)} chars. Summary: {res_str[:150]}...")
                                
                                # 针对预览图的特殊处理，以及流式传输特殊数据
                                if "preview_base64" in result_dict:
                                    yield SSEUtils.images([result_dict["preview_base64"]])
                                    # 不要将 base64 放入记忆，太大了
                                    clean_res = result_dict.copy()
                                    del clean_res["preview_base64"]
                                    memory.add_message("tool", content=json.dumps(clean_res), tool_call_id=meta["call"].id)
                                
                                # 适配 AudioTranscribeTool 的前端流式事件
                                elif "corrected_text" in result_dict:
                                    text = result_dict["corrected_text"]
                                    # Send text event for frontend display
                                    yield SSEUtils.format_event("audio_text_start", "")
                                    yield SSEUtils.format_event("audio_text_chunk", text)
                                    
                                    # Send violation event if any
                                    if "violation_check" in result_dict:
                                        v_data = result_dict["violation_check"]
                                        if v_data.get("is_violation"):
                                            frontend_data = {
                                                "is_violation": True,
                                                "time_anchors": v_data.get("segments", [])
                                            }
                                            yield SSEUtils.violation(frontend_data)
                                    
                                    memory.add_message("tool", content=json.dumps(result_dict), tool_call_id=meta["call"].id)

                                else:
                                    memory.add_message("tool", content=json.dumps(result_dict), tool_call_id=meta["call"].id)

                else:
                    # 最终回答
                    final_content = ai_message.content or ""
                    yield SSEUtils.log("📝 正在生成最终报告...")
                    print(f"--- [Final Report] ---\n{final_content}\n----------------------")
                    yield SSEUtils.format_event("final_report_start", "")
                    yield SSEUtils.token(final_content)
                    yield SSEUtils.format_event("final_report_end", "")
                    break

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield SSEUtils.error(f"致命错误: {str(e)}")
                break