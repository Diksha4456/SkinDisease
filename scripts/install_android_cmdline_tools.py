import urllib.request
import zipfile
import os
import shutil

url = "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"
sdk_dir = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk")
cmdline_dir = os.path.join(sdk_dir, "cmdline-tools")
latest_dir = os.path.join(cmdline_dir, "latest")
os.makedirs(cmdline_dir, exist_ok=True)
dest_zip = os.path.join(sdk_dir, "cmdline-tools.zip")

print("Downloading official Google Android Command-Line Tools...")
urllib.request.urlretrieve(url, dest_zip)
print("Download complete. Extracting...")

temp_extract = os.path.join(sdk_dir, "cmdline_temp")
os.makedirs(temp_extract, exist_ok=True)
with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
    zip_ref.extractall(temp_extract)

if os.path.exists(dest_zip):
    os.remove(dest_zip)

# The zip contains a folder named "cmdline-tools"
extracted_cmdline = os.path.join(temp_extract, "cmdline-tools")
if os.path.exists(latest_dir):
    shutil.rmtree(latest_dir)

shutil.move(extracted_cmdline, latest_dir)
if os.path.exists(temp_extract):
    shutil.rmtree(temp_extract)

sdkmanager_path = os.path.join(latest_dir, "bin", "sdkmanager.bat")
if os.path.exists(sdkmanager_path):
    print(f"SUCCESS: sdkmanager is installed at: {sdkmanager_path}")
else:
    print("Extraction finished, but sdkmanager.bat not found.")
