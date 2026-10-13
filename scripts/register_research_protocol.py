import json
from pathlib import Path
from trafficflow.research_eval import register_protocol

if __name__ == '__main__':
    value = json.loads(Path('config/research_protocol_v2.json').read_text(encoding='utf-8'))
    result = register_protocol(Path('reports/research/protocol_v2_registered.json'), value)
    print(result['protocol_sha256'])
