@echo off
rem Build remora-llama for Windows: CPU-only, static CRT (/MT), single portable exe.
rem Fixes the 0xC0000135 loader issue on Win10 boxes (no VC runtime dependency).
set LOG=C:\models\build_remora_mt.log
echo BUILD_MT_START %date% %time% >> %LOG%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >> %LOG% 2>&1
cd /d C:\models\llama-cpp-remora
git checkout remora-trace >> %LOG% 2>&1
git log -1 --oneline >> %LOG% 2>&1
echo CONFIGURE >> %LOG%
cmake -B build-mt -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_CPU=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded >> %LOG% 2>&1
echo BUILDING >> %LOG%
cmake --build build-mt --target llama-server -j 8 >> %LOG% 2>&1
echo BUILD_DONE %date% %time% >> %LOG%
dir build-mt\bin\llama-server.exe >> %LOG% 2>&1
