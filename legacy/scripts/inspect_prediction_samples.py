import json
import sys

for name in sys.argv[1:]:
    data = json.load(open(name))
    print(name, type(data).__name__, len(data) if hasattr(data, "__len__") else "")
    print(data[0] if isinstance(data, list) and data else data)
