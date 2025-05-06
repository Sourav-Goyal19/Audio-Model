import os
import shutil
import tempfile
from speech_predict import audio_predict
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/x-flac",
    "audio/flac",
    "audio/webm",
    "audio/mp4"
}

@app.get('/')
def home():
    return {"message": "hi"}


@app.post('/audio-prediction')
async def audio_prediction(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(content={
            "error": "missing audio file",
            "success": False,
            "prediction": None,
            "filename": None,
        }, status_code=400)

    if file.content_type not in ALLOWED_AUDIO_TYPES:
        return JSONResponse(content={
            "error": f"Invalid file type: {file.content_type}. Must be an audio file.",
            "success": False,
            "prediction": None,
            "filename": file.filename,
        }, status_code=415)
        
    try:
        suffix = os.path.splitext(file.filename)[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        prediction = audio_predict(temp_path)

        return JSONResponse(content={
            "prediction": prediction,
            "filename": file.filename,
            "success": True,
            "error": None
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={
            "prediction": None,
            "error": str(e),  
            "filename": file.filename,
            "success": False
        }, status_code=500)

    finally:
        await file.close()
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)