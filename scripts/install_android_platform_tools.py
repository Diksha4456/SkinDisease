import urllib.request
import zipfile
import os

url = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
sdk_dir = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk")
os.makedirs(sdk_dir, exist_ok=True)
dest_zip = os.path.join(sdk_dir, "platform-tools.zip")

print("Downloading official Google Android Platform-Tools (adb)...")
urllib.request.urlretrieve(url, dest_zip)
print("Download complete. Extracting...")

with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
    zip_ref.extractall(sdk_dir)

if os.path.exists(dest_zip):
    os.remove(dest_zip)

adb_path = os.path.join(sdk_dir, "platform-tools", "adb.exe")
if os.path.exists(adb_path):
    print(f"SUCCESS: adb.exe is installed at: {adb_path}")
else:
    print("Extraction finished, but adb.exe not found.")
