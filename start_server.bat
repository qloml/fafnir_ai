@@echo off
echo Starting Fafnir Server and Spectator GUI...

start "Fafnir Server" cmd /k "venv\Scripts\uvicorn.exe fast_server_0424:socket_app --host 0.0.0.0 --port 8765"

timeout /t 2 /nobreak > nul

start "ngrok" cmd /k "ngrok http 8765" 
start "Spectator GUI" venv\Scripts\python.exe clients\spectator_gui.py --name uo

echo Done!
