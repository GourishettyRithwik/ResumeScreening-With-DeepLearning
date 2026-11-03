# Resume Screener - Stop All Microservices (PowerShell)
Write-Host ""
Write-Host "Stopping all Resume Screener services..." -ForegroundColor Yellow
Stop-Process -Name "node","java","python" -ErrorAction SilentlyContinue
Write-Host "All services stopped." -ForegroundColor Green
Write-Host ""
