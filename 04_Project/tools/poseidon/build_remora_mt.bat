@echo off
rem Build remora-llama for Windows: CPU-only, static CRT (/MT), portable AVX2 target.
rem  - single portable exe, no VC runtime dependency (fixes 0xC0000135 on clean Win10)
rem  - GGML_NATIVE=OFF + AVX2/FMA/F16C only: runs on Skylake+; DO NOT leave
rem    GGML_NATIVE=ON (it bakes the BUILD machine's ISA — e.g. AVX-512 on
rem    Tiger Lake — and crashes with 0xC000001D illegal instruction on older
rem    CPUs like Poseidon's i7-6700HQ)
set LOG=C:\models\build_remora_mt.log
echo BUILD_MT_START %date% %time% >> %LOG%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >> %LOG% 2>&1
cd /d C:\models\llama-cpp-remora
git checkout remora-trace >> %LOG% 2>&1
git log -1 --oneline >> %LOG% 2>&1
echo CONFIGURE >> %LOG%
cmake -B build-mt -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DGGML_CPU=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded >> %LOG% 2>&1
echo BUILDING >> %LOG%
cmake --build build-mt --target llama-server -j 8 >> %LOG% 2>&1
echo BUILD_DONE %date% %time% >> %LOG%
dir build-mt\bin\llama-server.exe >> %LOG% 2>&1
