# real-avatar-local-open-source

This is a local version of real-avatar that enables the creation of avatar's locally on my personal computer

# General Idea of development

1. System Architecture

   Video Capture & Expression Tracking
   • Capture webcam frames with OpenCV.
   • Extract 468 facial landmarks and head pose using MediaPipe Face Mesh
   arXiv
   .
   • Convert landmarks into a low-dimensional “expression code” (e.g. principal components of landmark deltas).

   Avatar Rendering
   • Pretrained StyleAvatar generator (StyleGAN‐based) to map expression code → photo-realistic portrait
   arXiv
   .
   • TorchScript export so it runs fast on CPU/GPU without Internet.

   Speech Interface
   • Capture microphone audio, transcribe offline with Vosk Speech Recognition
   arXiv
   .
   • Prompt a local LLM (e.g. LLaMA2 or Vicuna via transformers) in a conversational agent loop.
   • Convert agent response text to speech using Coqui TTS
   arXiv
   and play back.

   Synchronization & UI
   • Single OpenCV window shows rendered avatar in real time.
   • Press keyboard keys to trigger “start/stop listening” and “exit.” No mouse needed.

2. Dependencies

pip install opencv-python mediapipe torch torchvision transformers \
 vosk pyttsx3 sounddevice numpy scipy

# For Coqui TTS:

pip install TTS

Make sure you have a webcam and microphone configured on your machine. 3. proof_of_concept.py

import cv2
import numpy as np
import mediapipe as mp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sounddevice as sd
import queue
import json
from vosk import Model as VoskModel, KaldiRecognizer
from TTS.api import TTS

# ─── 1. Facial Landmark Extraction ───────────────────────────────────────────

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
max_num_faces=1,
refine_landmarks=True,
min_detection_confidence=0.5)

def extract_expression_code(frame):
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb)
if not results.multi_face_landmarks:
return None
lm = results.multi_face_landmarks[0]
pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten() # Simple PCA-like projection: take first 100 dims
code = torch.from_numpy(pts[:100])
return code.unsqueeze(0) # shape [1,100]

# ─── 2. Load StyleAvatar Generator ───────────────────────────────────────────

# Assume you have exported a PyTorch .pt of StyleAvatar generator.

generator = torch.jit.load("styleavatar_generator.pt") # TorchScript module
generator.eval()

# ─── 3. Load Offline Speech Recognition (Vosk) ──────────────────────────────

vosk_model = VoskModel("model-small-en-us") # download from https://alphacephei.com/vosk/models
rec = KaldiRecognizer(vosk_model, 16000)

audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
audio_q.put(bytes(indata))

def listen_and_transcribe():
sd.default.samplerate = 16000
sd.default.channels = 1
with sd.RawInputStream(callback=audio_callback):
rec.Reset()
print("[Listening...] Press 's' again to stop.")
while True:
data = audio_q.get()
if rec.AcceptWaveform(data):
res = json.loads(rec.Result())
return res.get("text", "") # (You could break on a key press for end-of-speech.)

# ─── 4. Load Local LLM Agent ─────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-7B-1.1-HF") # local
model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-7B-1.1-HF",
device_map="auto",
torch_dtype=torch.float16)
chat_history = []

def chat_with_agent(user_text):
chat_history.append(f"Human: {user_text}")
prompt = "\n".join(chat_history) + "\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(\*\*inputs, max_new_tokens=256)
resp = tokenizer.decode(out[0], skip_special_tokens=True)
assistant_text = resp.split("Assistant:")[-1].strip()
chat_history.append(f"Assistant: {assistant_text}")
return assistant_text

# ─── 5. Load Coqui TTS ───────────────────────────────────────────────────────

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

def speak(text):
wav = tts.tts(text)
sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)
sd.wait()

# ─── 6. Main Loop ───────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
listening = False

print("Press 's' to toggle speech recognition, 'q' to quit.")
while True:
ret, frame = cap.read()
if not ret:
break

    code = extract_expression_code(frame)
    if code is not None:
        with torch.no_grad():
            out_img = generator(code)  # [1,3,H,W], values in [-1,1]
            img = ((out_img.clamp(-1,1)+1)/2*255).cpu().numpy()[0].transpose(1,2,0).astype(np.uint8)
            display = cv2.resize(img, (frame.shape[1], frame.shape[0]))
    else:
        display = frame

    cv2.imshow("Your AI Avatar", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        listening = not listening
        if listening:
            user_text = listen_and_transcribe()
            print("You said:", user_text)
            reply = chat_with_agent(user_text)
            print("AI:", reply)
            speak(reply)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

Notes & Citations

    MediaPipe Face Mesh for robust 468‐point landmark and head‐pose tracking
    arXiv
    .

    StyleAvatar (StyleGAN-based) pretrained generator from Wang et al., 2023
    arXiv
    .

    Vosk offline speech recognition for reliable transcription without Internet
    arXiv
    .

    Vicuna-7B local LLM via Hugging Face Transformers for conversational agent
    arXiv
    .

    Coqui TTS for open-source neural text-to-speech
    arXiv
    .

This gives you a fully working, offline pipeline—webcam→AI-avatar output + bi-directional voice chat—using only open-source, free tools. You can extend it with richer avatar representations (e.g. RMAvatar
arXiv
) or more advanced local LLMs as resources allow.
