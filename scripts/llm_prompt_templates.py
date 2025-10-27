"""
Prompt templates for LLM-based quality prediction, explanation and corrective actions.
This module only builds prompt strings; it does not call any external API.
"""
from typing import Dict, Any

BASE_INSTRUCTIONS = (
    "You are an expert injection-molding process analyst. "
    "Given a single production cycle's sensor readings and model outputs, perform the following tasks:\n"
    "1) Classify product quality as HIGH / MEDIUM / LOW (probabilistic judgment).\n"
    "2) Identify any likely defect type(s) (short shot, burn mark, sink/void, flash, warp) and explain why (link to sensors).\n"
    "3) Provide a concise chain-of-thought reasoning (3-6 short steps) describing how you reached the conclusion.\n"
    "4) Provide a confidence score between 0.0 and 1.0 and a short justification for that score.\n"
    "5) Suggest immediate corrective actions (1-3 actionable steps) prioritized by impact and safety.\n"
    "Return output as JSON with keys: quality, defects, reasoning, confidence, corrective_actions, suggestions_for_measurements."
)

EXAMPLE_JSON_SCHEMA = {
    "quality": "HIGH|MEDIUM|LOW",
    "defects": [
        {"type": "short_shot|burn|sink|flash|warp|none", "reason": "sensor-based reason text"}
    ],
    "reasoning": ["step 1","step 2"],
    "confidence": 0.85,
    "corrective_actions": ["increase mold temp by X deg","reduce injection speed"],
    "suggestions_for_measurements": ["capture cavity image at camera A","log pressure trace for next 50 cycles"]
}


def build_quality_prompt(sensor_row: Dict[str, Any], model_outputs: Dict[str, Any] = None) -> str:
    """Builds a single prompt string for the LLM given sensor values (and optional model outputs).

    sensor_row: mapping of sensor_name -> value (e.g., 'OilTemperature': 34)
    model_outputs: optional mapping with keys like 'cluster_k3', 'anomaly_flag', 'anomaly_score', or predicted quality if available
    """
    header = BASE_INSTRUCTIONS + "\n\nRespond ONLY with JSON conforming to the schema below.\n\nSchema:\n" + str(EXAMPLE_JSON_SCHEMA) + "\n\n"
    # Add sensor table
    sensor_lines = [f"{k}: {v}" for k, v in sensor_row.items()]
    sensors_block = "\n".join(sensor_lines)

    model_block = ""
    if model_outputs:
        model_lines = [f"{k}: {v}" for k, v in model_outputs.items()]
        model_block = "\n\nModel outputs:\n" + "\n".join(model_lines)

    final = header + "\nSensors:\n" + sensors_block + model_block + "\n\nNotes:\n- If you are uncertain, give a confidence < 0.6 and explain missing data.\n- Keep corrective actions short and actionable.\n"
    return final


if __name__ == '__main__':
    # quick manual check
    example_sensors = {"OilTemperature": 34, "MeasuredCycleDuration": 56.8, "MaxInjectionPressure": 71.23}
    print(build_quality_prompt(example_sensors, {"cluster_k3": 1, "anomaly_flag": -1, "anomaly_score": -0.2}))
