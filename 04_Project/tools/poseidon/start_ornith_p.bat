@echo off
rem Poseidon: Ornith-1.5-35B-A3B via OFFICIAL llama.cpp b10509 (Clang build, runs on Win10)
set SRV=C:\models\llama-cpp-b10509\llama-server.exe
set MODEL=C:\models\Ornith\Ornith-1.5-35B-Q4_K_M.gguf
if not exist %SRV% (
    echo SRV_NOT_FOUND >> C:\models\ornith_server.log
    exit /b 1
)
if not exist %MODEL% (
    echo MODEL_NOT_FOUND >> C:\models\ornith_server.log
    exit /b 1
)
echo SERVER_START_STOCK %date% %time% >> C:\models\ornith_server.log
start /b /d C:\models\llama-cpp-b10509 %SRV% -m %MODEL% --host 0.0.0.0 --port 11435 -c 65536 -ctk q8_0 -ctv q8_0 -ngl 0 --threads 8 --parallel 2 --api-key API_KEY_REDACTED >> C:\models\ornith_server.log 2>&1
echo SERVER_LAUNCHED %date% %time% >> C:\models\ornith_server.log
