import json
# import class_extractor
import section_extractor

# with open("payload/devops-engineer.json", encoding="utf-8") as f:
with open("sample_input_payload.json", encoding="utf-8") as f:
# with open("payload/payload_2409.json", encoding="utf-8") as f:
    test_input = json.load(f)

event = {"body": json.dumps(test_input)}
# sectionsv2.handler(event, {})
section_extractor.handler(event, {})
