import json
p='outputs/taguchi_runs/run_T1.ndjson'
with open(p,'r',encoding='utf-8') as f:
    line=f.readline()
    obj=json.loads(line)
print('PROMPT PREVIEW:\n', obj.get('prompt')[:800])
print('\nINTERNAL METADATA:\n', json.dumps(obj.get('internal_metadata'), indent=2, ensure_ascii=False))
print('\nRAW_ROW_REDACTED keys:', list(obj.get('raw_row_redacted', {}).keys()))
