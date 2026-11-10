from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import uvicorn, os, uuid, hashlib, secrets, cv2, time

from detector import TrafficDetector

# ─────────────────────────────────────────────
# Auth config
SECRET_KEY = "smarttraffic_secret_2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def hash_password(password: str) -> str:
    salt = "smarttraffic_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed

# In-memory user store (replace with DB in production)
fake_users: dict = {
    "admin": {
        "username": "admin",
        "email": "admin@traffic.ai",
        "hashed_password": hash_password("admin123"),
        "role": "admin",
    }
}

# In-memory comment/message store
comments: list = []
messages: list = []

# ─────────────────────────────────────────────
app = FastAPI(title="SmartTrafficAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

detector = TrafficDetector()

# ─────────────────────────────────────────────
# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    email: str
    role: str

class CommentIn(BaseModel):
    username: str
    text: str
    analysisId: Optional[str] = None

class MessageIn(BaseModel):
    sender: str
    receiver: str
    text: str

# ─────────────────────────────────────────────
# Auth helpers
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return fake_users[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ─────────────────────────────────────────────
# AUTH routes
@app.post("/api/auth/register", response_model=Token)
def register(user: UserCreate):
    if user.username in fake_users:
        raise HTTPException(status_code=400, detail="Username already exists")
    fake_users[user.username] = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hash_password(user.password),
        "role": "user",
    }
    token = create_access_token({"sub": user.username})
    return Token(access_token=token, token_type="bearer",
                 username=user.username, email=user.email, role="user")

@app.post("/api/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    u = fake_users.get(form.username)
    if not u or not verify_password(form.password, u["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": u["username"]})
    return Token(access_token=token, token_type="bearer",
                 username=u["username"], email=u["email"], role=u["role"])

@app.get("/api/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"], "email": current_user["email"], "role": current_user["role"]}

# ─────────────────────────────────────────────
# TRAFFIC routes
@app.get("/api/traffic/status")
def get_status():
    return detector.get_latest_data()

@app.get("/api/traffic/video_feed")
def video_feed():
    def generate():
        while True:
            frame = detector.get_frame()
            if frame is None:
                # If no frame, return a black image or small delay
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04) # roughly 25 fps
            
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/traffic/history")
def get_history():
    return detector.get_history()

@app.post("/api/traffic/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Only video files are accepted")
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = f"uploads/{filename}"
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)
    analysis_id = detector.process_video(filepath)
    return {"message": "Video uploaded. Processing started.", "analysisId": analysis_id, "filename": file.filename}

@app.post("/api/traffic/stop")
def stop_processing():
    detector.stop()
    return {"message": "Processing stopped"}

# ─────────────────────────────────────────────
# STATS route (aggregated from history)
@app.get("/api/stats")
def get_stats():
    hist = detector.get_history()
    if not hist:
        return {"totalAnalyses": 0, "avgVehicles": 0, "avgGreenTime": 10, "densityBreakdown": {}}
    density_count = {}
    for h in hist:
        d = h["density"]
        density_count[d] = density_count.get(d, 0) + 1
    avg_v = round(sum(h["avgVehicles"] for h in hist) / len(hist), 1)
    avg_g = round(sum(h["greenTime"] for h in hist) / len(hist), 1)
    return {
        "totalAnalyses": len(hist),
        "avgVehicles": avg_v,
        "avgGreenTime": avg_g,
        "densityBreakdown": density_count,
    }

# ─────────────────────────────────────────────
# COMMENTS routes
@app.get("/api/comments")
def get_comments():
    return comments

@app.post("/api/comments")
def add_comment(comment: CommentIn):
    c = {
        "id": str(uuid.uuid4())[:8],
        "username": comment.username,
        "text": comment.text,
        "analysisId": comment.analysisId,
        "createdAt": datetime.now().isoformat(),
    }
    comments.append(c)
    return c

@app.delete("/api/comments/{comment_id}")
def delete_comment(comment_id: str):
    global comments
    comments = [c for c in comments if c["id"] != comment_id]
    return {"message": "Deleted"}

# ─────────────────────────────────────────────
# MESSAGES routes
@app.get("/api/messages")
def get_messages():
    return messages

@app.post("/api/messages")
def send_message(msg: MessageIn):
    m = {
        "id": str(uuid.uuid4())[:8],
        "sender": msg.sender,
        "receiver": msg.receiver,
        "text": msg.text,
        "createdAt": datetime.now().isoformat(),
        "read": False,
    }
    messages.append(m)
    return m

# ─────────────────────────────────────────────
# EMERGENCY alert route
@app.post("/api/emergency")
def emergency_alert(body: dict):
    detector.current_data["signalStatus"] = "EMERGENCY"
    return {"message": f"Emergency override activated for: {body.get('type', 'unknown')}"}

# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
