# FairSkin AI — Flutter Mobile Application

This directory contains the production-ready **Flutter Mobile Application** for FairSkin AI.

## Architecture & Integration
* **Backend REST API**: The Flutter app communicates seamlessly with the local FairSkin AI backend running on Flask (`http://127.0.0.1:5000` or your machine's LAN IP / emulator IP `http://10.0.2.2:5000`).
* **Zero Accuracy Loss Guarantee**: The mobile application utilizes the exact same multi-task MobileNetV3 deep learning checkpoint (`models/best_fairskin_model.pth`), differential clinical prior calibration, and Star Schema Data Warehouse (`fairskin_dwh.db`).
* **Endpoints Used**:
  - `POST /api/screen` (Multipart/form-data image upload from Camera/Gallery)
  - `POST /api/screen_base64` (Alternative JSON base64 upload)
  - `GET /api/dwh/analytics` (Live Kimball Data Marts: fairness audits across FST I-VI and disease prevalence)
  - `GET /api/health` (Offline/Online handshake)

## How to Run

### 1. Ensure the Python Backend is Running
```powershell
# From project root
python app.py
```
Backend runs at `http://127.0.0.1:5000`.

### 2. Configure Backend Host in Flutter
In `lib/services/skin_screening_api.dart`:
* For **Windows / Desktop / Web**: Use `http://127.0.0.1:5000` (default)
* For **Android Emulator**: Use `http://10.0.2.2:5000`
* For **Physical Device (Phone over Wi-Fi)**: Use your computer's local IP (e.g., `http://192.168.1.15:5000`)

### 3. Run the Flutter App
```bash
cd mobile_flutter
flutter pub get
flutter run
```
