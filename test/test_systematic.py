import os
import sys
import importlib

# Mock API keys for testing (agent_config already has the real ones, but environment testing here)
os.environ["S2_API_KEY"] = "smKWRu1X8H3t7IkQSAaTk4OIc6Yni3F58nOEJtC6"

sys.path.append(os.path.dirname(__file__))

systematic = importlib.import_module("skills.03_academic.10_systematic")

print("Successfully imported tool.")
res = systematic.run_systematic_review("deep learning time series", limit_per_source=2)
print(res[:500])
