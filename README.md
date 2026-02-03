# Assessment OCR Scanner

A complete AI-powered web application that captures images from your webcam, extracts data from assessment sheets using OCR, and automatically stores the data in Google Sheets.

## Features

- 📷 **Live Webcam Capture** - Real-time webcam feed with capture functionality
- �️ **Real-Time OCR Mode** - Live text detection with bounding boxes and confidence visualization
- 📤 **Image Upload** - Drag & drop or browse to upload images
- 🔍 **AI-Powered OCR** - Extracts text and structured data from assessment sheets
- 📊 **Google Sheets Integration** - Automatically saves extracted data to Google Sheets
- 📈 **Statistics Dashboard** - Track total records, today's scans, and success rate
- 🔎 **Search Functionality** - Search records by registration number
- 🎨 **Modern UI** - Beautiful, responsive dark-themed interface
- 🧵 **Multi-threaded Processing** - Smooth video display with background OCR processing

## Real-Time OCR Features

The new Real-Time OCR mode provides production-level text detection based on multi-threaded architecture:

- **4 View Modes:**
  - Mode 1: High confidence only (>75%)
  - Mode 2: Color coded (green=high, red=low confidence)
  - Mode 3: Gradient (brightness = confidence level)
  - Mode 4: All detections
- **Adjustable crop region** for focused OCR area
- **Multi-language support** (English, Chinese, Japanese, Korean, etc.)
- **Live FPS counter** and performance monitoring
- **Bounding boxes** drawn around detected text in real-time

## Project Structure

```
OCR/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (credentials)
├── README.md              # This file
├── run_realtime_ocr.bat   # Standalone real-time OCR launcher
├── modules/
│   ├── __init__.py        # Package initialization
│   ├── realtime_ocr.py    # Production real-time OCR module (NEW)
│   ├── ocr_extractor.py   # OCR and data extraction module
│   ├── sheets_manager.py  # Google Sheets integration
│   └── webcam_capture.py  # Webcam handling module
├── templates/
│   └── index.html         # Web interface
└── uploads/               # Captured/uploaded images (auto-created)
```

## Prerequisites

1. **Python 3.8+** - Download from [python.org](https://python.org)
2. **Tesseract OCR** - Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Windows: Download installer from the link above
   - Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. **Google Cloud Service Account** - For Google Sheets access

## Installation

### Step 1: Install Python Dependencies

```bash
cd c:\Users\Admin\Desktop\OCR
pip install -r requirements.txt
```

### Step 2: Install Tesseract OCR

1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Note the installation path (default: `C:\Program Files\Tesseract-OCR`)
4. Update the path in `.env` file if different

### Step 3: Configure Google Sheets Access

1. The `.env` file already contains your service account credentials
2. Make sure your Google Sheet is shared with the service account email:
   - Email: `restaurant-bot-sheets@restaurant-bot-485706.iam.gserviceaccount.com`
   - Give it "Editor" access

### Step 4: Run the Application

```bash
python app.py
```

Open your browser and go to: **http://localhost:5000**

## Usage

### Using Webcam

1. Click **"Start Camera"** to activate your webcam
2. Position the assessment sheet in front of the camera
3. Click **"Capture & Scan"** or press **SPACE** to capture
4. Review the extracted data on the right panel
5. Click **"Save to Sheet"** to store in Google Sheets

### Using Image Upload

1. Drag & drop an image onto the upload area, or click to browse
2. Click **"Scan Uploaded Image"** to extract data
3. Review and save to Google Sheets

## Extracted Data Fields

The application extracts the following information from assessment sheets:

| Field | Description |
|-------|-------------|
| Registration No | Student's registration number |
| College Code | Institution code |
| Date | Assessment date |
| Degree/Branch | e.g., BE/ECE |
| Semester/Section | e.g., VI 'B' |
| Subject Code | e.g., CEC367 |
| Subject Name | e.g., Industrial IoT and Industry 4.0 |
| Part A Marks | Individual question marks (Q1-Q5) |
| Part B&C Marks | Detailed marks for questions 6-9 |
| Total Marks | Sum of all marks |
| Grand Total | Final score out of 100 |
| Course Outcomes | CO1, CO2 breakdown |
| Examiner Name | Name of the examiner |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/camera/start` | POST | Start webcam |
| `/api/camera/stop` | POST | Stop webcam |
| `/api/camera/stream` | GET | Stream webcam feed |
| `/api/capture-and-extract` | POST | Capture and extract data |
| `/api/upload-and-extract` | POST | Upload image and extract |
| `/api/save-to-sheet` | POST | Save data to Google Sheets |
| `/api/stats` | GET | Get statistics |
| `/api/recent-records` | GET | Get recent records |
| `/api/search` | POST | Search by registration |
| `/api/health` | GET | Health check |

### Real-Time OCR Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/realtime-ocr/start` | POST | Start real-time OCR |
| `/api/realtime-ocr/stop` | POST | Stop real-time OCR |
| `/api/realtime-ocr/stream` | GET | Stream with OCR overlay |
| `/api/realtime-ocr/text` | GET | Get current detected text |
| `/api/realtime-ocr/capture` | POST | Capture current frame |
| `/api/realtime-ocr/view-mode` | POST | Change view mode (1-4) |
| `/api/realtime-ocr/status` | GET | Get OCR status |

## Standalone Real-Time OCR

You can run the real-time OCR as a standalone desktop application:

```bash
# Run with defaults
python -m modules.realtime_ocr

# With custom options
python -m modules.realtime_ocr --view 2 --crop 150 100 --lang eng

# Using the batch file on Windows
run_realtime_ocr.bat
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --src` | Camera source index | 0 |
| `-v, --view` | View mode (1-4) | 1 |
| `-c, --crop` | Crop area (width height) | 100 100 |
| `-l, --lang` | OCR language code | eng |
| `--easyocr` | Use EasyOCR engine | False |
| `-r, --resolution` | Camera resolution | 1280 720 |

### Keyboard Controls (Standalone Mode)

| Key | Action |
|-----|--------|
| `c` | Capture current frame |
| `q` | Quit application |
| `1-4` | Change view mode |

## Troubleshooting

### Camera not working
- Make sure no other application is using the camera
- Try changing `CAMERA_INDEX` in config.py (0, 1, 2...)
- On Windows, ensure camera permissions are granted

### OCR not extracting text
- Ensure Tesseract is installed correctly
- Update `TESSERACT_PATH` in `.env` file
- Make sure the image is clear and well-lit
- Try holding the document steady

### Google Sheets connection failed
- Verify service account credentials in `.env`
- Ensure the sheet is shared with the service account email
- Check internet connection

## License

This project is for educational purposes.

## Support

For issues or questions, please check the troubleshooting section above.
