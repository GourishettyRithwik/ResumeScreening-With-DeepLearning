# Resume Screener - Start All Microservices (PowerShell)
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "   STARTING RESUME SCREENER MICROSERVICES PLATFORM" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Step 0: Kill any stale processes and clear Vite cache
Write-Host "[Cleanup] Stopping stale processes..." -ForegroundColor Yellow
Stop-Process -Name "node","java","python" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$viteCachePath = Join-Path $PSScriptRoot "react-frontend\node_modules\.vite"
if (Test-Path $viteCachePath) {
    Remove-Item -Recurse -Force $viteCachePath -ErrorAction SilentlyContinue
    Write-Host "[Cleanup] Cleared Vite cache." -ForegroundColor Yellow
}

# Step 1: Start Python ML Service (Port 5000)
Write-Host ""
Write-Host "[1/4] Starting Python ML Service (Port 5000)..." -ForegroundColor Green
$pythonJob = Start-Process -FilePath "python" -ArgumentList "ml_api.py" -WorkingDirectory (Join-Path $PSScriptRoot "python-ml-service") -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3

# Step 2: Start Java Parser Service (Port 8080)
Write-Host "[2/4] Starting Java Parser Service (Port 8080)..." -ForegroundColor Green
$javaJob = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", ".\gradlew.bat bootRun" -WorkingDirectory (Join-Path $PSScriptRoot "java-parser-service") -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 5

# Step 3: Start Node.js Gateway (Port 5001)
Write-Host "[3/4] Starting Node.js Gateway (Port 5001)..." -ForegroundColor Green
$nodeJob = Start-Process -FilePath "node" -ArgumentList "index.js" -WorkingDirectory (Join-Path $PSScriptRoot "node-gateway") -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2

# Step 4: Start React Frontend (Port 5173)
Write-Host "[4/4] Starting React Frontend (Port 5173)..." -ForegroundColor Green
$reactJob = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev" -WorkingDirectory (Join-Path $PSScriptRoot "react-frontend") -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3

# Step 5: Open Chrome
Write-Host ""
Write-Host "Opening Chrome..." -ForegroundColor Cyan
Start-Process "chrome" "http://localhost:5173"

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "   ALL SERVICES LAUNCHED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "   React Frontend:  http://localhost:5173" -ForegroundColor White
Write-Host "   Node.js Gateway: http://localhost:5001" -ForegroundColor White
Write-Host "   Java Parser:     http://localhost:8080" -ForegroundColor White
Write-Host "   Python ML API:   http://localhost:5000" -ForegroundColor White
Write-Host ""
Write-Host "   To stop all services, run: .\stop.ps1" -ForegroundColor Yellow
Write-Host "=====================================================" -ForegroundColor Cyan
