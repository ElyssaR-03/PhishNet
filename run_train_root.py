import traceback
from backend.train_models import train_phishnet_models

if __name__ == '__main__':
    try:
        print('Calling train_phishnet_models()...')
        detector, scores = train_phishnet_models()
        print('Training completed, scores:', scores)
    except Exception:
        print('Exception during training:')
        traceback.print_exc()
