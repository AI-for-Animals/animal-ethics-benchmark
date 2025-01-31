This folder contains files to run AHA (Animal Harm Assessment) benchmark.

Example use in google colab:

```
# 1 Clone the files
from google.colab import userdata
personal_token = userdata.get('GITHUB_TOKEN')
repository_url = "https://{token}@github.com/AI-for-Animals/animal-ethics-benchmark.git".format(token=personal_token)
!git clone --depth 1 --no-checkout https://github.com/AI-for-Animals/animal-ethics-benchmark.git temp
!cd temp && git sparse-checkout init --cone && git sparse-checkout set aha && git checkout
!mv temp/aha /content/ && rm -rf temp

# 2 Clone the files
import os
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
!pip install inspect-ai
!pip install anthropic
!pip install google-generativeai
!pip install openai

# 3 Run example batch with analysis
!python /content/aha/aha.py --model 'google/gemini-1.5-flash-002' --batch_size 2 --run-analysis
```
