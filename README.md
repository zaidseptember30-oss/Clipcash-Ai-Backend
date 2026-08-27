# ClipCash AI — automatic raw-video Shorts engine

Pipeline:
1. Upload raw video.
2. FFmpeg extracts mono 16 kHz audio.
3. OpenAI transcription creates timestamped segments.
4. AI selects the strongest moments using those timestamps.
5. FFmpeg renders each selection as a 1080x1920 vertical MP4.
6. The dashboard shows download links.

Requirements:
- Python 3.12+
- FFmpeg installed (or Docker)
- OPENAI_API_KEY environment variable

Run:
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"
python server.py

Open http://localhost:3000

Docker:
docker build -t clipcash-ai .
docker run -p 3000:3000 -e OPENAI_API_KEY="YOUR_KEY" clipcash-ai

Important:
- This version makes actual MP4 Shorts, but its captions are metadata/UI captions, not burned into the video.
- The next production step is word-level captions, face-aware 9:16 cropping, scene detection, storage, authentication, job queue and rate limits.
- Do not upload copyrighted material you don't have permission to process or redistribute.
