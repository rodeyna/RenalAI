from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, uuid
from app.model import predict

app = FastAPI(title='RenalAI DL Service')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'dl-microservice'}

@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    # Save uploaded image to a temp file
    tmp_path = f'/tmp/{uuid.uuid4()}_{file.filename}'
    with open(tmp_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict(tmp_path, seg_threshold=0.60)
        return {
            'diagnosis':          result['diagnosis'],
            'confidence_pct':     result['confidence_pct'],
            'all_probabilities':  result['all_probabilities'],
            'stone_coverage_pct': result['stone_coverage_pct'],
            'severity':           result['severity'],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)