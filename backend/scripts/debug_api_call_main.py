import sys, os, traceback
# Ensure the backend directory is on sys.path so `import main` resolves to backend/main.py
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
backend_dir = os.path.join(repo_root, 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post('/train', json={'n_samples': 100})
    print('status:', resp.status_code)
    print('text:', resp.text)
    try:
        print('json:', resp.json())
    except Exception:
        pass
except Exception:
    traceback.print_exc()
