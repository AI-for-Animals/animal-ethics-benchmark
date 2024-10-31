from google.colab import userdata
import os

# Install
#!pip install inspect-ai
#!pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
#!pip install openai
#!pip install anthropic
#!pip install --upgrade google-generativeai
#!pip install mistralai


# Set the API key
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['TOGETHER_API_KEY'] = userdata.get('TOGETHER_API_KEY')
os.environ['MISTRAL_API_KEY'] = userdata.get('MISTRAL_API_KEY')

#models
# https://inspect.ai-safety-institute.org.uk/
# https://docs.anthropic.com/en/docs/about-claude/models 
# https://ai.google.dev/gemini-api/docs/models/gemini 
# https://docs.mistral.ai/getting-started/models/models_overview/
# https://api.together.ai/models 


# Run the inspect command

#!inspect eval anai_inspect_task_v2.py --model anthropic/claude-3-haiku-20240307 --limit 1000
#!inspect eval anai_inspect_task_v2.py --model anthropic/claude-3-5-sonnet-20241022 --limit 1000
#!inspect eval anai_inspect_task_v2.py --model openai/gpt-4o-2024-08-06	--limit 1000
#!inspect eval anai_inspect_task_v2.py --model google/gemini-1.5-pro-002 --limit 1000
#!inspect eval anai_inspect_task_v2.py --model together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo --limit 1000
#!inspect eval anai_inspect_task_v2.py --model openai/o1-preview-2024-09-12 --limit 1000
!inspect eval anai_inspect_task_v2.py --model mistral/mistral-large-2407 --limit 1000
