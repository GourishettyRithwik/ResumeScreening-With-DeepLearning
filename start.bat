@echo off
echo =====================================================================
echo           STARTING RESUME SCREENER MICROSERVICES PLATFORM
echo =====================================================================
echo.

echo 1. Starting Python ML Microservice (Port 5000)...
start "Python ML Service" cmd /k "cd python-ml-service && python ml_api.py"
ping 127.0.0.1 -n 4 > nul

echo 2. Starting Java Parser Microservice (Port 8080)...
start "Java Parser Service" cmd /k "cd java-parser-service && .\gradlew.bat bootRun"
ping 127.0.0.1 -n 8 > nul

echo 3. Starting Node.js Express API Gateway (Port 5001)...
start "Node.js Gateway" cmd /k "cd node-gateway && npm start"
ping 127.0.0.1 -n 4 > nul

echo 4. Starting React.js Frontend (Port 5173)...
start "React.js Frontend" cmd /k "cd react-frontend && npm run dev"

echo.
echo =====================================================================
echo   All microservices launched!
echo.
echo   - React Frontend: http://localhost:5173
echo   - Node.js Gateway: http://localhost:5001
echo   - Java Parser: http://localhost:8080
echo   - Python ML API: http://localhost:5000
echo =====================================================================
