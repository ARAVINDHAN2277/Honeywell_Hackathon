import google.generativeai as genai

genai.configure(api_key="AIzaSyDcIQVxty2gLnIbFpgAf1a2IXmSLpQFeLY")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
