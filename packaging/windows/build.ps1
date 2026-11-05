# Builds the Windows installer and the portable ZIP.
# Run from the repository root in PowerShell:  .\packaging\windows\build.ps1
$ErrorActionPreference = "Stop"
$version = (python -c "import simon_enhance; print(simon_enhance.__version__)").Trim()

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
pyinstaller packaging\simon_enhance.spec --noconfirm --clean

# Portable: same bundle + marker file -> settings are stored next to the exe
$portable = "dist\SimonImageEnhance-$version-Windows-Portable"
if (Test-Path $portable) { Remove-Item $portable -Recurse -Force }
Copy-Item dist\SimonImageEnhance $portable -Recurse
Set-Content "$portable\portable.txt" "Portable mode: settings and edits are stored in the 'data' folder next to SimonImageEnhance.exe."
Compress-Archive -Path "$portable\*" -DestinationPath "$portable.zip" -Force

# Installer (needs Inno Setup 6: choco install innosetup)
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
& $iscc "/DMyAppVersion=$version" packaging\windows\installer.iss
Write-Host "Fertig: dist\SimonImageEnhance-Setup.exe und $portable.zip"
