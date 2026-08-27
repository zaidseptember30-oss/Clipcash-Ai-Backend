import os
import uuid
import json
import subprocess
import threading
import re
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# Allow your GitHub Pages frontend to communicate with Render
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/outputs/*": {"origins": "*"}
})

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not configured.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

JOBS = {}


# ============================================================
# HELPERS
# ============================================================

def run_command(command):
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if process.returncode != 0:
        raise RuntimeError(process.stderr[-4000:])

    return process.stdout


def safe_filename(name):
    name = Path(name).name
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name[:150]


def ffprobe_duration(path):
    result = run_command([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(path)
    ])

    return float(result.strip())


def update_job(job_id, **values):
    if job_id in JOBS:
        JOBS[job_id].update(values)


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_audio(audio_path):

    if not client:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured on the server."
        )

    with open(audio_path, "rb") as audio_file:

        response = client.audio.transcriptions.create(
            model=os.getenv(
                "OPENAI_TRANSCRIBE_MODEL",
                "whisper-1"
            ),
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []

    for segment in response.segments:

        segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip()
        })

    return segments


# ============================================================
# AI CLIP SELECTION
# ============================================================

def choose_clips(segments, niche, count, duration):

    if not client:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured on the server."
        )

    transcript = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        for s in segments
    )

    prompt = f"""
You are an expert viral short-form video editor.

Analyze the timestamped transcript below and select exactly
{count} strong, NON-OVERLAPPING moments.

Niche:
{niche or "general"}

Original video duration:
{duration:.1f} seconds.

Choose moments that have strong potential for TikTok,
YouTube Shorts, Instagram Reels and similar platforms.

Prioritize:

- Strong hooks
- Surprising moments
- Funny moments
- Emotional moments
- Controversial/debatable moments
- Useful information
- Strong opinions
- Stories with a payoff
- Moments that create curiosity
- Moments that make viewers want to watch until the end

Each clip should normally be between 15 and 60 seconds.

Use ONLY timestamps present in the transcript.

Do not invent dialogue.

Return ONLY valid JSON.

Format:

{{
  "clips": [
    {{
      "start": 10.0,
      "end": 45.0,
      "title": "Short title",
      "hook": "Short opening hook",
      "caption": "On-screen caption",
      "description": "Short social media description",
      "score": 92,
      "edit_notes": "Editing recommendation"
    }}
  ]
}}

TRANSCRIPT:

{transcript[:60000]}
"""

    response = client.responses.create(
        model=os.getenv(
            "OPENAI_MODEL",
            "gpt-5-mini"
        ),
        input=prompt
    )

    text = response.output_text.strip()

    # Remove accidental markdown fences
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    data = json.loads(text)

    clips = data.get("clips", [])

    if not clips:
        raise RuntimeError(
            "AI could not find suitable clips."
        )

    return clips[:count]


# ============================================================
# CREATE SRT CAPTIONS
# ============================================================

def create_srt(segments, start, end, srt_path):

    selected = []

    for segment in segments:

        seg_start = max(
            segment["start"],
            start
        )

        seg_end = min(
            segment["end"],
            end
        )

        if seg_end <= seg_start:
            continue

        selected.append({
            "start": seg_start - start,
            "end": seg_end - start,
            "text": segment["text"]
        })

    def timestamp(seconds):

        milliseconds = int(seconds * 1000)

        hours = milliseconds // 3600000
        milliseconds %= 3600000

        minutes = milliseconds // 60000
        milliseconds %= 60000

        seconds_int = milliseconds // 1000
        milliseconds %= 1000

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds_int:02d},"
            f"{milliseconds:03d}"
        )

    with open(srt_path, "w", encoding="utf-8") as file:

        for index, segment in enumerate(selected, 1):

            file.write(
                f"{index}\n"
                f"{timestamp(segment['start'])} --> "
                f"{timestamp(segment['end'])}\n"
                f"{segment['text']}\n\n"
            )


# ============================================================
# RENDER SHORT
# ============================================================

def make_clip(
    source,
    output,
    start,
    end,
    srt_path
):

    duration = max(
        1,
        end - start
    )

    subtitle_path = str(srt_path).replace(
        "\\",
        "/"
    ).replace(
        ":",
        "\\:"
    )

    # Scale to fill a 9:16 canvas while preserving
    # the original aspect ratio.
    video_filter = (
        "scale=1080:1920:"
        "force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "setsar=1,"
        f"subtitles='{subtitle_path}':"
        "force_style="
        "'FontName=Arial,"
        "FontSize=18,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=1,"
        "Alignment=2,"
        "MarginV=110'"
    )

    run_command([
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-t",
        str(duration),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output)
    ])


# ============================================================
# BACKGROUND WORKER
# ============================================================

def process_worker(
    job_id,
    source,
    niche,
    count
):

    audio_path = None

    try:

        update_job(
            job_id,
            status="working",
            progress=5,
            message="Preparing your video..."
        )

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration = ffprobe_duration(source)

        update_job(
            job_id,
            progress=10,
            message="Extracting audio..."
        )

        # ----------------------------------------------------
        # Audio extraction
        # ----------------------------------------------------

        audio_path = OUT / f"{job_id}.wav"

        run_command([
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path)
        ])

        update_job(
            job_id,
            progress=25,
            message="Transcribing your video..."
        )

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        segments = transcribe_audio(
            audio_path
        )

        update_job(
            job_id,
            progress=45,
            message="AI is finding your strongest moments..."
        )

        # ----------------------------------------------------
        # AI selection
        # ----------------------------------------------------

        clips = choose_clips(
            segments,
            niche,
            count,
            duration
        )

        results = []

        total = len(clips)

        # ----------------------------------------------------
        # Render clips
        # ----------------------------------------------------

        for index, clip in enumerate(clips):

            start = max(
                0,
                float(clip["start"])
            )

            end = min(
                duration,
                float(clip["end"])
            )

            if end <= start:
                continue

            srt_path = OUT / (
                f"{job_id}_{index + 1}.srt"
            )

            output_path = OUT / (
                f"{job_id}_short_{index + 1}.mp4"
            )

            create_srt(
                segments,
                start,
                end,
                srt_path
            )

            progress = 50 + int(
                45 * (index + 1) / total
            )

            update_job(
                job_id,
                progress=progress,
                message=(
                    f"Rendering Short "
                    f"{index + 1}/{total}..."
                )
            )

            make_clip(
                source,
                output_path,
                start,
                end,
                srt_path
            )

            results.append({
                "id": index + 1,
                "title": clip.get(
                    "title",
                    f"Short #{index + 1}"
                ),
                "hook": clip.get(
                    "hook",
                    ""
                ),
                "caption": clip.get(
                    "caption",
                    ""
                ),
                "description": clip.get(
                    "description",
                    ""
                ),
                "score": clip.get(
                    "score",
                    0
                ),
                "edit_notes": clip.get(
                    "edit_notes",
                    ""
                ),
                "start": round(start, 1),
                "end": round(end, 1),
                "duration": round(
                    end - start,
                    1
                ),
                "url": (
                    f"/outputs/"
                    f"{output_path.name}"
                )
            })

            srt_path.unlink(
                missing_ok=True
            )

        if not results:

            raise RuntimeError(
                "No Shorts were successfully rendered."
            )

        update_job(
            job_id,
            status="done",
            progress=100,
            message="Your Shorts are ready!",
            clips=results
        )

    except Exception as error:

        print(
            f"JOB {job_id} ERROR:",
            error
        )

        update_job(
            job_id,
            status="error",
            progress=100,
            message=str(error),
            clips=[]
        )

    finally:

        if audio_path:
            audio_path.unlink(
                missing_ok=True
            )

        source.unlink(
            missing_ok=True
        )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():

    return (
        "ClipCash AI Backend is running!"
    )


@app.get("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "ClipCash AI",
        "openai_configured": bool(
            OPENAI_API_KEY
        )
    })


@app.post("/api/process")
def process_video():

    try:

        video = request.files.get(
            "video"
        )

        if not video:

            return jsonify({
                "error":
                "No video uploaded."
            }), 400

        if not video.filename:

            return jsonify({
                "error":
                "The uploaded video has no filename."
            }), 400

        try:

            count = int(
                request.form.get(
                    "count",
                    "5"
                )
            )

        except ValueError:

            count = 5

        count = max(
            1,
            min(count, 10)
        )

        niche = request.form.get(
            "niche",
            "general"
        ).strip()

        job_id = str(
            uuid.uuid4()
        )

        filename = safe_filename(
            video.filename
        )

        source = OUT / (
            f"{job_id}_{filename}"
        )

        video.save(source)

        JOBS[job_id] = {
            "status": "queued",
            "progress": 2,
            "message": "Video queued.",
            "clips": []
        }

        thread = threading.Thread(
            target=process_worker,
            args=(
                job_id,
                source,
                niche,
                count
            ),
            daemon=True
        )

        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


@app.get("/api/jobs/<job_id>")
def get_job(job_id):

    job = JOBS.get(job_id)

    if not job:

        return jsonify({
            "status": "error",
            "progress": 100,
            "message": "Job not found.",
            "clips": []
        }), 404

    return jsonify(job)


@app.get("/outputs/<filename>")
def get_output(filename):

    return send_from_directory(
        OUT,
        filename,
        as_attachment=False
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "3000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
