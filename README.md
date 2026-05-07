================================================================
  AI Face Attendance System — Steps to Run from Scratch
================================================================
Project   : AI-Based Face Recognition Attendance System

PRE-REQUISITES
--------------
1. Python 3.10 or higher
   Download: https://www.python.org/downloads/
   Verify:   python --version

2. pip (comes with Python 3.10+)
   Verify:   pip --version

3. A working webcam connected to your computer(Even an integrated one works fine)

4. Git (optional, only if cloning from repository)
   Download: https://git-scm.com/downloads
   
5.To run the Entire project you just have to go inside the attendance_system directory
   and execute the run.py file and it will do everything.

NOTE: First-time setup downloads ~800 MB (PyTorch + FaceNet models).
      Ensure you have a stable internet connection.

================================================================
STEP 1 — EXTRACT THE PROJECT
================================================================

Unzip the file:
   ai_attendance_system.zip

You should see this folder structure:
   attendance_system/
   ├── app.py
   ├── run.py
   ├── requirements.txt
   ├── core/
   │   ├── database.py
   │   ├── face_engine.py
   │   ├── anti_spoofing.py
   │   └── export_utils.py
   ├── frontend/
   │   └── index.html
   ├── data/           (auto-created)
   ├── exports/        (auto-created)
   └── registered_faces/ (auto-created)

================================================================
STEP 2 — INSTALL DEPENDENCIES
================================================================

Open a terminal / command prompt inside the attendance_system folder:

   cd attendance_system

Install all Python packages:

   pip install -r requirements.txt

This installs:
   - torch, torchvision       (deep learning)
   - facenet-pytorch           (MTCNN + FaceNet models)
   - opencv-python             (camera & image processing)
   - flask, flask-cors         (REST API server)
   - pandas, openpyxl          (CSV/Excel export)
   - scikit-learn, scipy       (similarity math)

First install may take 5–15 minutes depending on internet speed.

================================================================
STEP 3 — START THE BACKEND SERVER
================================================================

Inside the attendance_system folder, run:

   python app.py

You should see:
   =======================================================
     AI Attendance System — Backend
     http://localhost:5000
   =======================================================

Keep this terminal open. The backend must stay running.

================================================================
STEP 4 — OPEN THE FRONTEND DASHBOARD
================================================================

Open your web browser (Chrome or Firefox recommended).

Navigate to this file:
   attendance_system/frontend/index.html

OR drag and drop index.html into your browser.

You will see the FaceID Attend dashboard.

If the dashboard says "Backend not running" — go back to Step 3.

================================================================
STEP 5 — REGISTER PERSONS
================================================================

1. Click "Persons" in the left sidebar
2. Click "Add Person" (top right)
3. Enter:
   - Full Name
   - Employee/Student ID
   - Department (optional)
4. Click "Save & Continue"
5. Allow webcam access when browser asks
6. Click "Capture Sample" 5 times
   - Look straight → slightly left → slightly right
   - Each capture takes ~1 second to process
7. Click "Done"
8. Repeat for all persons to register

Tip: Register in good lighting for best accuracy.
Tip: Each person needs at least 5 samples for reliable recognition.

================================================================
STEP 6 — CREATE AN ATTENDANCE SESSION
================================================================

1. Click "Live Feed" in the sidebar
2. Type a session name (e.g. "Morning Roll Call")
3. Click "+ Create Session"
4. You will see "ACTIVE SESSION" appear below the button

================================================================
STEP 7 — MARK ATTENDANCE
================================================================

1. On the Live Feed page, click "▶ Start Camera"
2. Allow webcam access if prompted
3. The camera feed will appear with:
   - Green box = recognised person → attendance being marked
   - Orange box = live but unregistered person
   - Red box    = photo/spoof detected → BLOCKED
4. Walk in front of the camera
5. Attendance is marked automatically when:
   - Liveness Score >= 60  (real face confirmed)
   - Photo Score    < 40   (not a photo/screen)
   - Match confidence >= 65%
   - Person not already marked in this session
6. "✓ Attendance Marked" badge appears on recognised faces

Anti-Spoofing Note:
   The system will REJECT phone photos or printed photos.
   You must physically be in front of the camera.
   The system requires natural micro-movement and blinking.

================================================================
STEP 8 — VIEW ATTENDANCE
================================================================

Dashboard:
   Shows today's unique attendees, present/absent counts,
   attendance rate. Auto-refreshes every 3 seconds.

Attendance Log:
   Full record of all marked attendance with timestamps,
   confidence scores and liveness scores.

Sessions:
   List of all sessions with start/end times.

================================================================
STEP 9 — EXPORT DATA
================================================================

1. Click "Export / Import" in the sidebar
2. Choose:
   - "Export Today's Attendance (CSV)"    → .csv file
   - "Export Today's Attendance (Excel)"  → .xlsx file
   - "Export All Persons (CSV)"           → persons list
   - "Export Full Report (Excel)"         → all sessions, all data

Files are saved to: attendance_system/exports/

================================================================
STEP 10 — IMPORT PERSONS IN BULK (Optional)
================================================================

Create a CSV file with these columns:
   name, employee_id, department, email

Example (persons.csv):
   name,employee_id,department,email
   John Doe,EMP001,CSE,john@svnit.ac.in
   Jane Smith,EMP002,IT,jane@svnit.ac.in

Then:
1. Go to Export / Import
2. Click "Choose File" under Import Persons
3. Select your CSV file
4. Persons are added automatically

================================================================
TROUBLESHOOTING
================================================================

Problem: "Cannot open camera"
Fix:     Try Camera 1 or Camera 2 from the dropdown
         Ensure no other app (Zoom, Teams) is using the camera

Problem: "No face detected during registration"
Fix:     Improve lighting — face front of camera
         Move closer (within 1 metre)

Problem: "Backend not running" toast on dashboard
Fix:     Run: python app.py in a terminal
         Ensure port 5000 is not blocked by firewall

Problem: Recognition not working
Fix:     Register more face samples (aim for 8-10 per person)
         Ensure good, consistent lighting

Problem: pip install fails
Fix:     Try: pip install torch --index-url https://download.pytorch.org/whl/cpu
         Then: pip install -r requirements.txt

Problem: Attendance not updating on Dashboard
Fix:     The dashboard auto-refreshes every 3 seconds
         Click "Refresh" button manually if needed
         Ensure a session is active (create one in Live Feed)

================================================================
TECHNOLOGY STACK SUMMARY
================================================================

Component         | Technology
------------------|---------------------------------------------
Face Detection    | MTCNN (facenet-pytorch)
Face Recognition  | FaceNet InceptionResnetV1 (VGGFace2)
Anti-Spoofing     | Custom 6-cue liveness (OpenCV + NumPy)
Database          | SQLite with WAL mode
Backend API       | Python Flask + Flask-CORS
Frontend          | HTML / CSS / Vanilla JavaScript
Data Export       | Pandas + OpenPyXL
Camera I/O        | OpenCV VideoCapture

================================================================
PROJECT FILES REFERENCE
================================================================

app.py              Main Flask server — all API endpoints
core/database.py    SQLite ORM — all DB operations
core/face_engine.py MTCNN detection + FaceNet recognition
core/anti_spoofing.py Multi-cue liveness detection
core/export_utils.py  CSV + Excel export/import
frontend/index.html   Full dashboard UI (no framework needed)
