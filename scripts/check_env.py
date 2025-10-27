import os
print('GEMINI_ENDPOINT set:', bool(os.environ.get('GEMINI_ENDPOINT')))
print('GEMINI_API_KEY set:', bool(os.environ.get('GEMINI_API_KEY')))
print('LLM_PROVIDER:', os.environ.get('LLM_PROVIDER'))
print('LLM_MODEL:', os.environ.get('LLM_MODEL'))
