import uvicorn
import shutil
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# ---------------------------
# UVR imports
# ---------------------------
from demucs.apply import apply_model
from demucs.pretrained import get_model
import torchaudio

# ---------------------------
# Server Setup
# ---------------------------
app = FastAPI(title="UVR Processor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

MODEL = get_model("htdemucs")   # best all-round model


# ---------------------------
# Helper Functions
# ---------------------------

def save_upload_to_disk(upload: UploadFile) -> Path:
    """
    Saves uploaded file to temp directory and returns the path.
    """
    file_id = str(uuid.uuid4())
    file_path = TEMP_DIR / f"{file_id}_{upload.filename}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    return file_path


def separate_stems(audio_path: Path, lane: str) -> dict:
    """
    Runs UVR / Demucs stem separation.
    Returns dict with paths to stems.
    """

    wav, sr = torchaudio.load(str(audio_path))

    model = MODEL
    stems = apply_model(model, wav, sr, device="cpu")

    # model.stems gives the names: ["drums", "bass", "other", "vocals"]
    output_dir = TEMP_DIR / f"out_{audio_path.stem}"
    output_dir.mkdir(exist_ok=True)

    stem_map = {}

    for name, audio in zip(model.sources, stems):
        out_path = output_dir / f"{name}.wav"
        torchaudio.save(str(out_path), audio, sr)
        stem_map[name] = str(out_path)

    # For ACCA lanes, return only vocals + instrumental
    if "acca" in lane:
        from pydub import AudioSegment

        vocal_path = stem_map.get("vocals")
        other_parts = [p for n, p in stem_map.items() if n != "vocals"]

        instrumental_mix = AudioSegment.silent(duration=0)

        for part in other_parts:
            instrumental_mix = instrumental_mix.overlay(AudioSegment.from_wav(part))

        inst_path = output_dir / "instrumental.wav"
        instrumental_mix.export(str(inst_path), format="wav")

        return {
            "vocals": vocal_path,
            "instrumental": str(inst_path)
        }

    return stem_map


# ---------------------------
# API Endpoint
# ---------------------------

@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    lane: str = Form(...)
):
    """
    Main processing endpoint.
    Takes uploaded file + lane string.
    Returns JSON with paths to stems.
    """
    # Save input file
    input_path = save_upload_to_disk(file)

    # Run UVR separation
    stems = separate_stems(input_path, lane)

    # Convert paths → downloadable URLs
    # For Render: static file serving must go through /files/
    result = {
        "stems": [
            {
                "name": name,
                "path": path,
                "url": f"/files/{os.path.basename(path)}"
            }
            for name, path in stems.items()
        ]
    }

    return result


# ---------------------------
# Static file serving
# ---------------------------
from fastapi.staticfiles import StaticFiles
app.mount("/files", StaticFiles(directory=str(TEMP_DIR)), name="files")


# ---------------------------
# Run locally
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
