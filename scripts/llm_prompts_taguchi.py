"""Taguchi-aligned prompt templates and generator for injection-molding quality tasks.

This module exposes a PromptGenerator class that fills templates based on
factor levels (A-D) and returns a dict with 'prompt' and metadata.

The templates are Turkish, designed to request strict JSON output when asked.
"""
from typing import Dict
import datetime

# Fields (sensor/measurement names) to include in the prompt measurements section.
# These come from the provided attachment and represent the key parameters we want
# to send to the LLM for each sample.
PARAM_FIELDS = [
    "InjectionStroke",
    "InjectionTime",
    "ActualStrokePosition",
    "MeasuredCycleDuration",
    "cluster3_flag",
    "SliderOutputTimePeriodValue",
    "MoldTemp2",
    "MaxInjectionPressure",
    "SliderInputTimePeriodValue",
    "CoolingTime",
    "OilTemperature",
    "DosingTime",
    "ClosingForceGenerationTimePeriodValue",
    "MoldTemp6",
    "BarrelTemp1",
]


class PromptGenerator:
    def __init__(self, schema_name: str = "taguchi_v1"):
        self.schema_name = schema_name

    def level_to_description(self, A: int, B: int, C: int, D: int) -> Dict[str, str]:
        # A: Context depth, B: COT, C: Output strictness, D: Persona
        ctx_map = {
            1: "son 3 okuma",
            2: "son 10 okuma ve kısa özet",
            3: "son 30 okuma ve özet istatistikler (ortalama/std)"
        }
        cot_map = {1: "COT kapalı", 2: "Kısa COT (2-4 adım)", 3: "Detaylı COT (6-12 adım)"}
        out_map = {1: "Serbest metin", 2: "Yarı yapılandırılmış (başlık + JSON)", 3: "Sıkı JSON (yalnızca JSON)"}
        persona_map = {1: "Nötr", 2: "Process Engineer", 3: "Quality Expert"}

        return {
            "context": ctx_map.get(A, "son 10 okuma"),
            "cot": cot_map.get(B, "COT kapalı"),
            "output": out_map.get(C, "Sıkı JSON"),
            "persona": persona_map.get(D, "Nötr")
        }

    def generate_prompt(self, sample: dict, A: int, B: int, C: int, D: int, prompt_id: str) -> Dict[str, object]:
        """Return a dict with filled prompt text and metadata.

        sample: dict must contain sample_id, MouldCode, timestamp, setpoints and a timeseries-summary string.
        """
        desc = self.level_to_description(A, B, C, D)

        system = (
            "You are a concise Quality Expert specialized in injection molding. "
            "When asked, output ONLY the requested JSON between triple backticks."
        )

        cot_instruction = {
            1: "Do not include chain-of-thought; reasoning_steps should be an empty array.",
            2: "Provide a brief 2-4 step chain-of-thought in reasoning_steps.",
            3: "Provide a detailed 6-12 step chain-of-thought in reasoning_steps."
        }[B]

        strict_json_note = "Sadece ve yalnızca triple-backticks içinde geçerli JSON döndürün. JSON dışında hiç bir metin yok." if C == 3 else "JSON ile birlikte kısa açıklama kabul edilir."

        # Explicit instruction: do not use timestamp or other metadata for inference
        # Tracking is handled externally; the model should only use Setpoints and Measurements
        # to make quality assessments and should not reference Timestamp or internal IDs in output.
        do_not_use_timestamp = (
            "Not: Lütfen 'Timestamp' veya diğer meta bilgileri kalite değerlendirmesi için kullanmayın; "
            "yalnızca 'Setpoints' ve 'Measurements' alanlarını kullanarak çıkarım yapın. "
            "Cevabınızda zaman damgası veya dahili kimlik bilgilerini belirtmeyin.\n\n"
        )

        prompt = (
            f"Sistem: {system}\n\n"
            f"Bağlam: Bu değerlendirme için {desc['context']} kullanın. {cot_instruction}\n"
            f"Çıktı formatı: {desc['output']}. Persona: {desc['persona']}. {strict_json_note}\n\n"
            f"Veri:\n"
            f"sample_id: {sample.get('sample_id')}\n"
            f"Setpoints: {sample.get('setpoints')}\n"
            f"Timeseries summary: {sample.get('timeseries_summary')}\n\n"
            f"{do_not_use_timestamp}"
            "Measurements:\n"
        )

        # Append selected measurement fields (from sample['measurements'] or sample['raw_row'])
        measurements = sample.get('measurements') or sample.get('raw_row') or {}
        if isinstance(measurements, dict):
            for k in PARAM_FIELDS:
                # Use empty string when value missing to avoid None literal in prompt
                v = measurements.get(k, '')
                prompt += f"{k}: {v}\n"
            prompt += "\n"

        # Conclude prompt with the expected JSON schema and strict-return instruction
        prompt += (
            "JSON şeması: sample_id, quality (High/Medium/Low), confidence (0..1), predicted_defects (array), reasoning_steps (array), recommended_actions (array), provenance.\n"
            "Cevabı sadece triple-backticks içinde geçerli JSON olarak verin.\n"
        )

        metadata = {
            "prompt_id": prompt_id,
            "A": A, "B": B, "C": C, "D": D,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "schema": self.schema_name
        }

        return {"system": system, "prompt": prompt, "metadata": metadata}


if __name__ == "__main__":
    # small local preview
    pg = PromptGenerator()
    s = {"sample_id":"row_1","MouldCode":5001,"timestamp":"2025-10-27T12:00:00Z",
         "setpoints":"mold_temp=60C,inj_pressure=120bar,cycle_time=8s",
         "timeseries_summary":"[...sampled 30 readings...]"}
    out = pg.generate_prompt(s, A=3, B=3, C=3, D=3, prompt_id="L9-T1")
    print(out['prompt'])
