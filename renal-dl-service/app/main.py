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
    # Save to /app to use the Docker Volume bridge
    file_path = f'/app/{file.filename}' 
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Pass the file_path to the model
        result = predict(file_path, seg_threshold=0.60)
        return {
            'diagnosis':          result['diagnosis'],
            'confidence_pct':     result['confidence_pct'],
            'all_probabilities':  result['all_probabilities'],
            'stone_coverage_pct': result['stone_coverage_pct'],
            'severity':           result['severity'],
            'result_image_path':  result.get('result_image_path', 'No image generated')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
  
        pass