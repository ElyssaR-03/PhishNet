from fastapi.testclient import TestClient
import traceback
import os
import sys

# Ensure repo root is on sys.path so 'backend' package imports resolve
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
backend_pkg = os.path.join(repo_root, 'backend')
if backend_pkg not in sys.path:
    sys.path.insert(0, backend_pkg)

try:
    from backend.main import app
    client = TestClient(app)
    resp = client.post('/train', json={'n_samples': 100})
    print('status:', resp.status_code)
    print('text:', resp.text)
    try:
        print('json:', resp.json())
    except Exception:
        pass
except Exception as e:
    traceback.print_exc()
