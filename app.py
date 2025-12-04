import io
import os
import sqlite3
from datetime import datetime, date, timedelta
import threading
import logging
import re
import csv
from io import StringIO
import time
import base64

from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try insightface imports
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    insightface = None
    FaceAnalysis = None

# Config
DB_PATH = "attendance.db"
FACE_DIR = "data/faces"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# ---------------------- DB Initialization ----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT,
        course TEXT,
        embedding BLOB,
        created_at TEXT,
        photos_count INTEGER DEFAULT 0
    )""")
    
    # Create attendance table WITHOUT unique constraint first
    c.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        name TEXT,
        course TEXT,
        timestamp TEXT,
        date TEXT
    )""")
    
    # Create user_photos table for multiple photos per user
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        photo_path TEXT,
        embedding BLOB,
        created_at TEXT
    )""")
    
    # Check if date column exists in attendance table
    c.execute("PRAGMA table_info(attendance)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'date' not in columns:
        print("Adding 'date' column to attendance table...")
        c.execute("ALTER TABLE attendance ADD COLUMN date TEXT")
        
        # Migrate existing data: extract date from timestamp
        c.execute("SELECT id, timestamp FROM attendance WHERE date IS NULL")
        rows = c.fetchall()
        for row_id, timestamp in rows:
            if timestamp:
                try:
                    # Try to extract date from timestamp
                    if 'T' in timestamp:
                        date_part = timestamp.split('T')[0]
                    else:
                        date_part = timestamp[:10]
                    c.execute("UPDATE attendance SET date = ? WHERE id = ?", (date_part, row_id))
                except:
                    pass
    
    # Now add unique constraint if it doesn't exist
    try:
        # Check if constraint already exists
        c.execute("PRAGMA index_list(attendance)")
        indexes = [idx[1] for idx in c.fetchall()]
        
        if 'idx_attendance_user_date' not in indexes:
            # Create unique constraint
            c.execute("CREATE UNIQUE INDEX idx_attendance_user_date ON attendance(user_id, date)")
    except:
        pass
    
    # Create other indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_course ON users(course)")
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

# ---------------------- DB Functions ----------------------
def save_user_with_multiple_photos(user_id: str, name: str, course: str, embeddings: list):
    """Save user with multiple photos for better accuracy"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Clean name
    clean_name = ' '.join([word.capitalize() for word in name.split()])
    now = datetime.now().isoformat()
    
    # Save main user record (average embedding)
    avg_embedding = np.mean(embeddings, axis=0)
    emb_blob = avg_embedding.astype(np.float32).tobytes()
    
    c.execute("""
        INSERT OR REPLACE INTO users (id, name, course, embedding, created_at, photos_count) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, clean_name, course, emb_blob, now, len(embeddings)))
    
    # Save individual photos
    for i, emb in enumerate(embeddings):
        photo_path = f"{FACE_DIR}/{user_id}_photo_{i+1}.jpg"
        photo_emb_blob = emb.astype(np.float32).tobytes()
        c.execute("""
            INSERT INTO user_photos (user_id, photo_path, embedding, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, photo_path, photo_emb_blob, now))
    
    conn.commit()
    conn.close()
    return clean_name

def get_all_users():
    """Get all users with their embeddings"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, course, embedding FROM users ORDER BY name")
    rows = c.fetchall()
    users = []
    for r in rows:
        uid, name, course, emb_blob = r
        emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
        users.append({"id": uid, "name": name, "course": course, "embedding": emb})
    conn.close()
    return users

def get_user_photos_embeddings(user_id: str):
    """Get all photo embeddings for a user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT embedding FROM user_photos WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    embeddings = []
    for r in rows:
        emb = np.frombuffer(r[0], dtype=np.float32).copy()
        embeddings.append(emb)
    conn.close()
    return embeddings

def add_attendance_if_not_exists(user_id: str, name: str, course: str = ""):
    """Add attendance only if not already marked for today"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get current date in YYYY-MM-DD format
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().isoformat()
    
    # Check if already marked today
    c.execute("SELECT id FROM attendance WHERE user_id = ? AND date = ?", (user_id, today))
    existing = c.fetchone()
    
    if existing:
        conn.close()
        return False, "Already marked today"
    
    # If course not provided, get it from users table
    if not course:
        c.execute("SELECT course FROM users WHERE id = ?", (user_id,))
        result = c.fetchone()
        course = result[0] if result else ""
    
    # Add new attendance
    c.execute("""
        INSERT INTO attendance (user_id, name, course, timestamp, date) 
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, name, course, current_time, today))
    
    conn.commit()
    conn.close()
    return True, "Attendance marked"

def delete_attendance(attendance_id: int):
    """Delete attendance record by ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
    conn.commit()
    conn.close()
    return c.rowcount > 0

def delete_user(user_id: str):
    """Delete user and all related data"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Delete from attendance
    c.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
    
    # Delete from user_photos
    c.execute("DELETE FROM user_photos WHERE user_id = ?", (user_id,))
    
    # Delete from users
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    
    conn.commit()
    conn.close()
    
    # Delete face images
    import glob
    for file in glob.glob(f"{FACE_DIR}/{user_id}*.jpg"):
        try:
            os.remove(file)
        except:
            pass
    
    return True

def list_attendance(limit=100):
    """Get attendance records"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, user_id, name, course, timestamp 
        FROM attendance 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def list_users():
    """Get all registered users"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, course, photos_count, created_at FROM users ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return rows

def get_stats():
    """Get system statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Total users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    
    # Today's attendance
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
    today_attendance = c.fetchone()[0]
    
    # Total attendance records
    c.execute("SELECT COUNT(*) FROM attendance")
    total_attendance = c.fetchone()[0]
    
    conn.close()
    
    return {
        "total_users": total_users,
        "today_attendance": today_attendance,
        "total_attendance": total_attendance
    }

def generate_user_id(name: str, course: str):
    """Generate user ID based on name and course"""
    # Get course code
    course_codes = {
        "employee": "EMP",
        "bca1": "BCA1",
        "bca2": "BCA2",
        "bca3": "BCA3"
    }
    
    prefix = course_codes.get(course, "STU")
    
    # Get initials from name
    words = name.strip().split()
    if len(words) >= 2:
        initials = (words[0][0] + words[-1][0]).upper()
    else:
        initials = name[:2].upper() if len(name) >= 2 else name.upper()
    
    # Get count for this course
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE course = ?", (course,))
    count = c.fetchone()[0] + 1
    conn.close()
    
    # Format: PREFIX_INITIALS_001
    return f"{prefix}_{initials}_{count:03d}"

# ---------------------- Face Recognition ----------------------
_face_analyzer = None
_face_lock = threading.Lock()

def get_face_analyzer():
    global _face_analyzer
    if _face_analyzer is not None:
        return _face_analyzer
    if FaceAnalysis is None:
        raise RuntimeError("InsightFace not installed.")
    with _face_lock:
        if _face_analyzer is None:
            logger.info("Loading InsightFace...")
            fa = FaceAnalysis(providers=["CPUExecutionProvider"])
            fa.prepare(ctx_id=-1, det_size=(640, 640))
            _face_analyzer = fa
            logger.info("InsightFace ready.")
    return _face_analyzer

def pil_from_bytes(data):
    return Image.open(io.BytesIO(data)).convert("RGB")

def get_embedding(img):
    arr = np.asarray(img)
    fa = get_face_analyzer()
    faces = fa.get(arr)
    if not faces:
        raise ValueError("No face detected")
    return np.array(faces[0].embedding, dtype=np.float32)

def get_multiple_embeddings(images):
    """Get embeddings from multiple images"""
    embeddings = []
    for img in images:
        emb = get_embedding(img)
        embeddings.append(emb)
    return embeddings

def l2_norm(x):
    x = x.astype(np.float32)
    n = np.linalg.norm(x)
    return x if n == 0 else x / n

def find_best_match(query_emb, users, threshold=0.5):
    """Find best match with all user photos"""
    q = l2_norm(query_emb)
    best, best_score = None, -1
    
    for u in users:
        # Get all photos for this user
        photo_embeddings = get_user_photos_embeddings(u["id"])
        
        # If no individual photos, use main embedding
        if not photo_embeddings:
            emb = l2_norm(u["embedding"])
            score = float(np.dot(q, emb))
        else:
            # Use best score from all photos
            scores = []
            for photo_emb in photo_embeddings:
                emb = l2_norm(photo_emb)
                score = float(np.dot(q, emb))
                scores.append(score)
            score = max(scores) if scores else 0.0
        
        if score > best_score:
            best = u
            best_score = score
    
    if best_score < threshold:
        return None, best_score
    return best, best_score

# ---------------------- Auto Attendance Detection ----------------------
class AutoAttendanceDetector:
    def __init__(self, check_interval=3):
        self.last_check_time = 0
        self.check_interval = check_interval
        self.last_detected_user = None
        self.detection_count = 0
        self.detection_threshold = 3  # Require 3 consecutive detections
        
    def should_check(self):
        current_time = time.time()
        if current_time - self.last_check_time >= self.check_interval:
            self.last_check_time = current_time
            return True
        return False
    
    def process_detection(self, user_id, score):
        if user_id == self.last_detected_user:
            self.detection_count += 1
        else:
            self.last_detected_user = user_id
            self.detection_count = 1
        
        # If detected multiple times consecutively, mark attendance
        if self.detection_count >= self.detection_threshold and score >= 0.6:
            return True
        return False

auto_detector = AutoAttendanceDetector()

# ---------------------- Routes ----------------------
@app.route("/")
def home():
    return redirect(url_for("webcam_page"))

@app.route("/register-page")
def register_page():
    return render_template("register.html")

@app.route("/webcam-page")
def webcam_page():
    return render_template("webcam.html")

@app.route("/attendance-list")
def attendance_list():
    try:
        attendance = list_attendance()
        return render_template("attendance_list.html", attendance=attendance)
    except Exception as e:
        return f"Error loading attendance: {str(e)}", 500

@app.route("/users-list")
def users_list():
    try:
        users = list_users()
        return render_template("users_list.html", users=users)
    except Exception as e:
        return f"Error loading users: {str(e)}", 500

# ---------------------- API Routes ----------------------
@app.route("/api/register", methods=["POST"])
def api_register():
    try:
        # Get form data
        name = request.form.get("name", "").strip()
        course = request.form.get("course", "").strip()
        
        # Get multiple photos from base64 data
        photos_data = request.form.getlist("photos[]")
        
        if not photos_data or len(photos_data) == 0:
            return jsonify({"error": "No photos uploaded"}), 400
        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not course:
            return jsonify({"error": "Course is required"}), 400
        
        # Generate user ID
        user_id = generate_user_id(name, course)
        
        # Process multiple images and get embeddings
        embeddings = []
        
        for i, photo_data in enumerate(photos_data):
            if i >= 5:  # Limit to 5 photos
                break
                
            try:
                # Convert base64 to image
                if ',' in photo_data:
                    photo_data = photo_data.split(',')[1]
                
                img_data = base64.b64decode(photo_data)
                img = pil_from_bytes(img_data)
                emb = get_embedding(img)
                embeddings.append(emb)
                
                # Save image
                photo_path = f"{FACE_DIR}/{user_id}_photo_{i+1}.jpg"
                img.save(photo_path)
            except ValueError as e:
                # Skip images without faces
                continue
            except Exception as e:
                logger.error(f"Error processing photo {i+1}: {str(e)}")
                continue
        
        if len(embeddings) == 0:
            return jsonify({"error": "No faces detected in any photos"}), 400
        
        # Save user with multiple photos
        clean_name = save_user_with_multiple_photos(user_id, name, course, embeddings)
        
        return jsonify({
            "status": "ok", 
            "id": user_id, 
            "name": clean_name,
            "course": course,
            "photos_count": len(embeddings)
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/mark", methods=["POST"])
def api_mark():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No photo uploaded"}), 400
        
        # Process image
        img = pil_from_bytes(file.read())
        arr = np.asarray(img)
        
        # Get face analyzer
        fa = get_face_analyzer()
        
        # Detect all faces in the image
        faces = fa.get(arr)
        
        if not faces:
            return jsonify({"status": "no_faces", "score": 0.0}), 400
        
        # Get all users
        users = get_all_users()
        if not users:
            return jsonify({"error": "No registered users found"}), 404
        
        results = []
        marked_count = 0
        
        for face in faces:
            # Get embedding for this face
            emb = np.array(face.embedding, dtype=np.float32)
            
            # Find best match
            match, score = find_best_match(emb, users, threshold=0.5)
            
            if match:
                # Try to mark attendance
                success, message = add_attendance_if_not_exists(
                    match["id"], 
                    match["name"], 
                    match.get("course", "")
                )
                
                results.append({
                    "user_id": match["id"],
                    "name": match["name"],
                    "score": float(score) if score is not None else 0.0,
                    "marked": success,
                    "message": message
                })
                
                if success:
                    marked_count += 1
            else:
                results.append({
                    "user_id": None,
                    "name": "Unknown",
                    "score": float(score) if score is not None else 0.0,
                    "marked": False,
                    "message": "No match found"
                })
        
        if len(results) == 1:
            # Single face - return simple response
            result = results[0]
            if result["user_id"]:
                return jsonify({
                    "status": "ok",
                    "user_id": result["user_id"],
                    "name": result["name"],
                    "score": result["score"],
                    "marked": result["marked"],
                    "message": result["message"]
                })
            else:
                return jsonify({
                    "status": "not_matched", 
                    "score": result["score"] if result["score"] is not None else 0.0
                })
        else:
            # Multiple faces
            return jsonify({
                "status": "multiple",
                "count": len(results),
                "marked_count": marked_count,
                "results": results
            })
            
    except ValueError as e:
        return jsonify({"status": "no_faces", "score": 0.0}), 400
    except Exception as e:
        logger.error(f"Mark attendance error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/auto-detect", methods=["POST"])
def api_auto_detect():
    """API for auto-detection from video stream"""
    try:
        # Get base64 image data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400
        
        # Convert base64 to image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_data = base64.b64decode(image_data)
        img = pil_from_bytes(img_data)
        
        # Get embedding
        emb = get_embedding(img)
        
        # Get all users
        users = get_all_users()
        if not users:
            return jsonify({"status": "no_users"})
        
        # Find best match
        match, score = find_best_match(emb, users, threshold=0.5)
        
        if match and score >= 0.6:
            # Check auto-attendance
            if auto_detector.process_detection(match["id"], score):
                # Mark attendance if not already marked today
                success, message = add_attendance_if_not_exists(
                    match["id"], 
                    match["name"], 
                    match.get("course", "")
                )
                
                if success:
                    return jsonify({
                        "status": "marked",
                        "user_id": match["id"],
                        "name": match["name"],
                        "score": float(score),
                        "message": "Attendance auto-marked"
                    })
                else:
                    return jsonify({
                        "status": "already_marked",
                        "user_id": match["id"],
                        "name": match["name"],
                        "score": float(score),
                        "message": message
                    })
            
            return jsonify({
                "status": "detected",
                "user_id": match["id"],
                "name": match["name"],
                "score": float(score),
                "detection_count": auto_detector.detection_count
            })
        
        return jsonify({"status": "no_match", "score": float(score)})
        
    except ValueError:
        return jsonify({"status": "no_face"})
    except Exception as e:
        logger.error(f"Auto detect error: {str(e)}")
        return jsonify({"status": "error"})

@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        stats = get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({"total_users": 0, "today_attendance": 0})

@app.route("/api/delete-attendance/<int:attendance_id>", methods=["DELETE"])
def api_delete_attendance(attendance_id):
    try:
        success = delete_attendance(attendance_id)
        if success:
            return jsonify({"status": "ok", "message": "Attendance deleted"})
        else:
            return jsonify({"error": "Attendance not found"}), 404
    except Exception as e:
        logger.error(f"Delete attendance error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/delete-user/<user_id>", methods=["DELETE"])
def api_delete_user(user_id):
    try:
        success = delete_user(user_id)
        if success:
            return jsonify({"status": "ok", "message": "User deleted"})
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logger.error(f"Delete user error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/attendance-list", methods=["GET"])
def api_attendance_list():
    try:
        attendance = list_attendance()
        
        # Convert to list of dicts
        records = []
        for record in attendance:
            records.append({
                "id": record[0],
                "user_id": record[1],
                "name": record[2],
                "course": record[3],
                "timestamp": record[4]
            })
        
        # Get stats
        stats = get_stats()
        
        return jsonify({
            "status": "ok",
            "records": records,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Attendance list error: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/api/export-attendance", methods=["GET"])
def export_attendance():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT user_id, name, course, timestamp FROM attendance ORDER BY timestamp DESC")
        rows = c.fetchall()
        conn.close()
        
        # Create CSV
        si = StringIO()
        writer = csv.writer(si)
        
        # Write header
        writer.writerow(['Student ID', 'Name', 'Course', 'Date', 'Time'])
        
        # Write data
        for row in rows:
            user_id, name, course, timestamp = row
            
            # Format course name
            course_names = {
                'employee': 'Employee',
                'bca1': 'BCA 1st Year',
                'bca2': 'BCA 2nd Year',
                'bca3': 'BCA 3rd Year'
            }
            course_display = course_names.get(course, course)
            
            # Split timestamp
            if 'T' in timestamp:
                date_part, time_part = timestamp.split('T')
                time_part = time_part[:5]
            else:
                date_part = timestamp[:10]
                time_part = timestamp[11:16] if len(timestamp) > 10 else ''
            
            writer.writerow([user_id, name, course_display, date_part, time_part])
        
        output = si.getvalue()
        
        # Create filename
        filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return output, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename={filename}'
        }
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users-list", methods=["GET"])
def api_users_list():
    try:
        users = list_users()
        
        # Convert to list of dicts
        user_list = []
        for user in users:
            user_list.append({
                "id": user[0],
                "name": user[1],
                "course": user[2],
                "photos_count": user[3],
                "created_at": user[4]
            })
        
        return jsonify({
            "status": "ok",
            "users": user_list,
            "count": len(user_list)
        })
        
    except Exception as e:
        logger.error(f"Users list error: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

# ---------------------- Main ----------------------
if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Print initial stats
    print("\n" + "="*50)
    print("Smart Attendance System")
    print("="*50)
    print("Features:")
    print("1. Auto-attendance detection (no button click needed)")
    print("2. Multiple photos per user (5 photos auto-captured)")
    print("3. Prevent duplicate daily attendance")
    print("4. Delete attendance/users functionality")
    print("5. Accurate time tracking")
    print("6. Auto 5-photo capture during registration")
    print("="*50)
    
    if insightface is None:
        print("WARNING: InsightFace not installed. Face recognition will not work.")
        print("Install with: pip install insightface")
    else:
        print("✓ InsightFace is available")
    
    print(f"\nServer running at: http://localhost:9000")
    print("Press Ctrl+C to stop\n")
    
    app.run(host="0.0.0.0", port=9000, debug=True)