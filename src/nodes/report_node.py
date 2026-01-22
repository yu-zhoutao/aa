import json
from typing import Optional
from .base_node import BaseNode, LogCallback
from ..state.state import JudgeState

SYSTEM_PROMPT_REPORT = """
# Role: 资深广电内容安全审查专家（北京局专项）

# Mission: 
你正在执行一项严格的内容合规审查任务。请严格基于**《北京局审核需求说明》**，对上传的内容进行多维度、全方位的研判。审核结论需清晰、准确，直接供领导决策参考。

# Context Data:
(以下是系统检测到的原始数据)
{data}

---

# Review Standards (北京局专项审核红线库):

请逐帧、逐句排查以下 **12 项核心风险点**（必须严格对应以下条款）：

# ## (A) 政治与意识形态（重中之重）
# * **(一) 民族、宗教信仰表现失当**：严禁极端宗教组织。
# * **(二) 政治体制和意识形态内容表现失当**：
#     * 严禁污蔑共产主义制度、诋毁/调侃/丑化国家领导人。
#     * 严禁台独倾向（如“青天白日满地红”旗帜）。
#     * 严禁对白中提取出反动文字。
# * **(三) 国际关系、涉外交**：严禁诋毁友好国家领导人（如普京、金正恩等），严禁诋毁社会主义制度。

# ## (B) 导向与价值观
# * **(四) 过度展现社会、人性阴暗面**：严禁宣扬血腥暴力、价值导向颓废（丧文化）、校园暴力、青少年吸毒贩毒。
# * **(六) 存在低俗涉性**：严禁大尺度低俗涉性镜头、大篇幅同性恋情节、卖腐内容。
# * **(七) 血腥暴力、涉毒**：严禁展现血腥暴力全过程、吸毒涉毒全过程。

# (十一) 煽动、策划非法集会、游行、示威。例如拉横幅，大规模人员聚集示威。
# (十二) 危害国家安全、颠覆国家政权，破坏国家统一。例如达赖、东突、藏独、疆独、台独。
# (十三) 泄露警务工作秘密：视频或图片中出现警察执法

特别注意：
- 若【网络事实核查】或【系统检测】中明确提示发现了"劣迹艺人"或"硬性政治红线"（如台独旗帜），必须**无条件直接判定违规**。
- 存在涉毒、暴力或阴暗面的视频，若主旨是弘扬警察正义、打击犯罪、弘扬正能量、揭露社会问题但最终导向光明，或通过反面教材警示观众的视频不属于违规视频，属于合规视频。
---

# Output Format (输出要求):

请严格按照以下 Markdown 格式输出（**不要输出开场白，语言需专业、客观**）：、

### 1. 内容摘要
(简述视频/音频/图片的核心事件，讲清楚“谁、在什么地方、做了什么”)

### 2. 事实与背景核查
* **信源核实**：(结合检索结果判断)
* **背景补充**：(简要补充事件背景，若无则填“无”)

### 3. 风险详细研判
(请逐条排查。**注意：仅输出存在风险的维度。若某维度合规，请务必直接跳过，不要在报告中提及。** 若全片均无风险，请输出“✅ 全片未发现明显违规内容”。)

**政治/意识形态**：(仅在有风险时输出)
* **违规点**：摘录具体违规画面或文字
* **违反条款**：条款名称
* **违规解析**：一句话解释为什么违规

**价值观/导向**：(仅在价值观/导向有风险时输出，格式同上)

**制作/历史/技术**：(仅在制作/历史/技术有风险时输出，格式同上)

### 4. 最终审核结论
* **判定结果**：合规 / 需人工复审 / 违规（硬性红线） 
* **核心理由**：(用一句话概括最致命的问题。若合规，此项可省略)
* **违规证据链**：(若合规，此项可省略)
    * **涉及条款**：完整条款名称
    * **具体证据**：原文/画面描述，并且说明具体在哪里体现了

### 5. 修改/处置建议
(给出具体操作建议，如“建议全网下架”、“建议删除xx秒至xx秒片段”)
""".strip()

class ReportNode(BaseNode):
    async def run(self, state: JudgeState, on_event: Optional[LogCallback] = None) -> JudgeState:
        await self.log_info("正在生成最终报告...", on_event)
        
        # 准备数据
        tasks_data = []
        for t in state.tasks:
            task_summary = {
                "dimension": t.dimension,
                "results": []
            }
            for r in t.results:
                result_lite = {
                    "tool": r.tool_name,
                    "finding": r.finding,
                    "is_violation": r.is_violation
                }
                if r.tool_name == "face_identify" and isinstance(r.raw_output, dict):
                    result_lite["persons"] = r.raw_output.get("persons", [])
                if r.tool_name == "ocr_detect" and isinstance(r.raw_output, dict):
                     ocr_items = r.raw_output.get("ocr_results", [])
                     texts = [item.get("text", "") for sub in ocr_items for item in sub.get("items", [])]
                     result_lite["ocr_sample"] = "".join(texts)[:200]
                
                # 添加 web_search 结果
                if r.tool_name == "web_search" and isinstance(r.raw_output, dict):
                     result_lite["search_findings"] = r.raw_output.get("search_findings", "")

                task_summary["results"].append(result_lite)
            tasks_data.append(task_summary)

        # 还要把预处理阶段的 shared_context 加进去，尤其是搜索结果
        # 因为 web_search 可能是在预处理阶段跑的，不在 tasks 里
        context_summary = {}
        if "face_identify_result" in state.shared_context:
            context_summary["face"] = state.shared_context["face_identify_result"]
        # web_search 结果目前在 shared_context 里没有显式保存，这是个遗漏点
        # 但我们在 PreProcessingNode 里虽然跑了 web_search，但没有把它放进 shared_context
        # 让我去 PreProcessingNode 补一下

        data_str = json.dumps({"tasks": tasks_data, "pre_check": context_summary}, ensure_ascii=False, indent=2)
        
        if on_event:
            await on_event("final_report_start", "")

        response = await self.llm_client.ainvoke(SYSTEM_PROMPT_REPORT, data_str)
        
        print("\n" + "="*20 + " [模型输出: 最终报告] " + "="*20)
        print(response)
        print("=" * 60 + "\n")
        
        if on_event:
            chunk_size = 5
            for i in range(0, len(response), chunk_size):
                await on_event("token", response[i:i+chunk_size])
                await asyncio.sleep(0.01)

            await on_event("final_report_end", "")

        state.final_report = response
        state.is_violation = "违规" in response or "不合规" in response
        state.is_completed = True
        
        return state