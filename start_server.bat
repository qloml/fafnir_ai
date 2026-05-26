@@echo off
echo Starting Fafnir Server and Spectator GUI...

start "Fafnir Server" cmd /k "uv run python -m uvicorn fast_server_0424:socket_app --host 0.0.0.0 --port 8765"

timeout /t 2 /nobreak > nul

start "ngrok" cmd /k "ngrok http 8765" 
start "Spectator GUI" uv run clients\spectator_gui.py --name uo

echo Done!
