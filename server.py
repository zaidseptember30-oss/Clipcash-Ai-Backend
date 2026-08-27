import os, uuid, json, subprocess, shutil, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from openai import OpenAI

BASE=Path(__file__).resolve().parent
OUT=BASE/"outputs"; OUT.mkdir(exist_ok=True)
app=Flask(__name__,static_folder="public")
CORS(app)
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JOBS={}

def run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-3000:])
    return p.stdout

def ffprobe_duration(path):
    return float(run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(path)]).strip())

def transcribe(audio):
    with open(audio,"rb") as f:
        r=client.audio.transcriptions.create(model=os.getenv("OPENAI_TRANSCRIBE_MODEL","whisper-1"),file=f,response_format="verbose_json",timestamp_granularities=["segment"])
    return [{"start":float(x.start),"end":float(x.end),"text":x.text.strip()} for x in r.segments]

def choose_clips(segments,niche,count,duration):
    transcript="\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    prompt=f"""You are an expert short-form editor. Select {count} non-overlapping moments from this timestamped transcript.
Niche: {niche or 'general'}; source duration: {duration:.1f}s.
Prioritize surprising, funny, emotional, useful or debate-worthy moments with a clear payoff. Each clip should be 15-60 seconds when possible.
Return ONLY valid JSON: {{"clips":[{{"start":number,"end":number,"title":string,"hook":string,"caption":string,"score":number,"edit_notes":string}}]}}
Use timestamps from the transcript. Do not invent dialogue.
TRANSCRIPT:
{transcript[:60000]}"""
    r=client.responses.create(model=os.getenv("OPENAI_MODEL","gpt-5-mini"),input=prompt)
    data=json.loads(r.output_text)
    return data["clips"]

def make_clip(src,out,start,end):
    # Center-crop to 9:16, preserve source audio, and burn a clean caption later.
    dur=max(1,end-start)
    vf="scale=1080:-2,crop=1080:1920:(iw-1080)/2:(ih-1920)/2"
    run(["ffmpeg","-y","-ss",str(start),"-i",str(src),"-t",str(dur),"-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])

def worker(job_id,src,niche,count):
    try:
        JOBS[job_id]={"status":"working","progress":10,"message":"Extracting audio...","clips":[]}
        audio=OUT/f"{job_id}.wav"
        run(["ffmpeg","-y","-i",str(src),"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(audio)])
        JOBS[job_id].update(progress=30,message="Transcribing with timestamps...")
        duration=ffprobe_duration(src); segs=transcribe(audio)
        JOBS[job_id].update(progress=55,message="AI is finding the strongest moments...")
        picks=choose_clips(segs,niche,count,duration)
        results=[]
        for i,c in enumerate(picks):
            start=max(0,float(c["start"])); end=min(duration,float(c["end"]))
            out=OUT/f"{job_id}_short_{i+1}.mp4"
            JOBS[job_id].update(progress=55+int(35*(i+1)/len(picks)),message=f"Rendering Short {i+1}/{len(picks)}...")
            make_clip(src,out,start,end)
            results.append({**c,"start":round(start,1),"end":round(end,1),"url":f"/outputs/{out.name}"})
        JOBS[job_id].update(status="done",progress=100,message="All Shorts are ready.",clips=results)
        audio.unlink(missing_ok=True); src.unlink(missing_ok=True)
    except Exception as e:
        JOBS[job_id].update(status="error",progress=100,message=str(e))

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/process")
def process():
    video=request.files.get("video")
    if not video: return jsonify(error="No video uploaded."),400
    job=str(uuid.uuid4()); src=OUT/f"{job}_{video.filename}"
    video.save(src)
    JOBS[job]={"status":"queued","progress":2,"message":"Queued...","clips":[]}
    threading.Thread(target=worker,args=(job,src,request.form.get("niche",""),int(request.form.get("count","5"))),daemon=True).start()
    return jsonify(job_id=job)

@app.get("/api/jobs/<job>")
def job(job):
    return jsonify(JOBS.get(job,{"status":"error","progress":100,"message":"Job not found.","clips":[]}))

@app.get("/outputs/<name>")
def output(name): return send_from_directory(OUT,name,as_attachment=False)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","3000")))
