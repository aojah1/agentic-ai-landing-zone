from __future__ import annotations

import asyncio
import time

from rich.console import Console

from agents import Runner, custom_span, gen_trace_id, trace, set_tracing_disabled

from src.agents.research_bot.planner import WebSearchItem, WebSearchPlan, planner_agent
from src.agents.research_bot.search import search_agent
from src.agents.research_bot.writer import ReportData, writer_agent
from src.utils.printer import Printer

# --- add imports at top if missing ---
import json, re
from pydantic import TypeAdapter

# --- helper: turn anything (str/dict) into WebSearchPlan or fail clearly ---
def _to_websearchplan(obj) -> WebSearchPlan:
    adapter = TypeAdapter(WebSearchPlan)

    if isinstance(obj, WebSearchPlan):
        return obj
    if isinstance(obj, dict):
        return adapter.validate_python(obj)

    # Coerce to str
    text = obj if isinstance(obj, str) else str(obj or "")

    # 1) strict JSON first
    try:
        return adapter.validate_python(json.loads(text))
    except Exception:
        pass

    # 2) grab first {...} block
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return adapter.validate_python(json.loads(text[s:e+1]))
        except Exception:
            pass

    # 3) regex for {"searches":[...]} block
    m = re.search(r'\{[^{}]*"searches"\s*:\s*\[.*?\][^{}]*\}', text, flags=re.S)
    if m:
        return adapter.validate_python(json.loads(m.group(0)))

    # 4) hard fail with snippet to debug prompt/response_format
    snippet = text[:300].replace("\n", " ")
    raise ValueError(f"Planner returned non-JSON or wrong shape. Snippet: {snippet}")

class ResearchManager:
    def __init__(self):
        self.console = Console()
        self.printer = Printer(self.console)
        set_tracing_disabled(True)

    async def run(self, query: str) -> None:
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            self.printer.update_item(
                "trace_id",
                f"Trace ID: {trace_id}",
                is_done=True,
                hide_checkmark=True,
            )

            self.printer.update_item(
                "starting",
                "Starting research...",
                is_done=True,
                hide_checkmark=True,
            )
            search_plan = await self._plan_searches(query)
            search_results = await self._perform_searches(search_plan)
            report = await self._write_report(query, search_results)

            final_report = f"Report summary\n\n{report.short_summary}"
            self.printer.update_item("final_report", final_report, is_done=True)

            self.printer.end()

        print("\n\n=====REPORT=====\n\n")
        print(f"Report: {report.markdown_report}")
        print("\n\n=====FOLLOW UP QUESTIONS=====\n\n")
        follow_up_questions = "\n".join(report.follow_up_questions)
        print(f"Follow up questions: {follow_up_questions}")

    # async def _plan_searches(self, query: str) -> WebSearchPlan:
    #     self.printer.update_item("planning", "Planning searches...")
    #     result = await Runner.run(
    #         planner_agent,
    #         f"Query: {query}",
    #     )
    #     self.printer.update_item(
    #         "planning",
    #         f"Will perform {len(result.final_output.searches)} searches",
    #         is_done=True,
    #     )
    #     return result.final_output_as(WebSearchPlan)

    # --- replace your _plan_searches with this ---
    async def _plan_searches(self, query: str) -> WebSearchPlan:
        self.printer.update_item("planning", "Planning searches...")
        result = await Runner.run(planner_agent, f"Query: {query}")

        # Prefer library parse; but it may return str in your build — handle both.
        plan_candidate = None
        try:
            plan_candidate = result.final_output_as(WebSearchPlan)
        except Exception:
            plan_candidate = result.final_output  # likely a str

        plan = _to_websearchplan(plan_candidate)

        self.printer.update_item(
            "planning",
            f"Will perform {len(plan.searches)} searches",
            is_done=True,
        )
        return plan


    async def _perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        with custom_span("Search the web"):
            self.printer.update_item("searching", "Searching...")
            num_completed = 0
            tasks = [asyncio.create_task(self._search(item)) for item in search_plan.searches]
            results = []
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
                num_completed += 1
                self.printer.update_item(
                    "searching", f"Searching... {num_completed}/{len(tasks)} completed"
                )
            self.printer.mark_item_done("searching")
            return results

    async def _search(self, item: WebSearchItem) -> str | None:
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def _write_report(self, query: str, search_results: list[str]) -> ReportData:
        self.printer.update_item("writing", "Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = Runner.run_streamed(
            writer_agent,
            input,
        )
        update_messages = [
            "Thinking about report...",
            "Planning report structure...",
            "Writing outline...",
            "Creating sections...",
            "Cleaning up formatting...",
            "Finalizing report...",
            "Finishing report...",
        ]

        last_update = time.time()
        next_message = 0
        async for _ in result.stream_events():
            if time.time() - last_update > 5 and next_message < len(update_messages):
                self.printer.update_item("writing", update_messages[next_message])
                next_message += 1
                last_update = time.time()

        self.printer.mark_item_done("writing")
        return result.final_output_as(ReportData)
