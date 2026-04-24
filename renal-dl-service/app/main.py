from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
import shutil, os, uuid
from app.model import predict

app = FastAPI(title='RenalAI DL Service')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# This line allows the browser to access images at http://localhost:8001/results/
app.mount("/results", StaticFiles(directory="/app"), name="results")

@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    # Save to /app so the static mount can find it
    file_path = f'/app/{uuid.uuid4()}_{file.filename}'
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = predict(file_path, seg_threshold=0.60)
        
        # Get just the filename (e.g., "abc_result.jpg") from the full path
        image_filename = os.path.basename(result.get('result_image_path', ''))
        
        # Construct the URL for the frontend
        image_url = f'http://localhost:8001/results/{image_filename}' if image_filename else None

        return {
            'diagnosis':          result['diagnosis'],
            'confidence_pct':     result['confidence_pct'],
            'all_probabilities':  result['all_probabilities'],
            'stone_coverage_pct': result['stone_coverage_pct'],
            'severity':           result['severity'],
            'mask_image_url':     image_url # <--- Frontend will use this!
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
