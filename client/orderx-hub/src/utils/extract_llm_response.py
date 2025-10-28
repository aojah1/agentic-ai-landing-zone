def extract_final_answer_from_chat_result(response_obj):
    try:
        chat_result = response_obj.data  # This is a ChatResult object
        traces = getattr(chat_result, "traces", [])
        for trace in traces:
            if getattr(trace, "trace_type", None) == "PLANNING_TRACE":
                output = getattr(trace, "output", "")
                match = re.search(r'"action":\s*"Final Answer".*?"action_inputs":\s*"([^"]+)"', output, re.DOTALL)
                if match:
                    return match.group(1).strip()
        return "❌ Final answer not found in planning trace."
    except Exception as e:
        return f"❌ Error: {e}"