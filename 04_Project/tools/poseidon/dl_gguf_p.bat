@echo off
rem Ornith Q4_K_M download for Poseidon (resumable via curl -C -)
if not exist C:\models\Ornith mkdir C:\models\Ornith
curl.exe -L --retry 8 --retry-delay 5 -C - -o C:\models\Ornith\Ornith-1.5-35B-Q4_K_M.gguf "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-Q4_K_M.gguf" >> C:\models\dl_gguf_p.log 2>&1
echo DL_GGUF_DONE %date% %time% >> C:\models\dl_gguf_p.log
