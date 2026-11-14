@echo off
rem Builds the embedding index and runs both evaluations with Ollama on this machine.
rem Progress is written to results\run.log
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
if not exist results mkdir results
set LOG=results\run.log
echo [%time%] start > %LOG%
if not exist .venv python -m venv .venv >> %LOG% 2>&1
call .venv\Scripts\activate.bat
python -m pip install -q -r requirements.txt >> %LOG% 2>&1
echo [%time%] pulling models (bge-m3, qwen2.5:7b) >> %LOG%
echo Downloading models, this window can stay in the background...
ollama pull bge-m3 >> %LOG% 2>&1
ollama pull qwen2.5:7b >> %LOG% 2>&1
echo [%time%] 1/3 index >> %LOG%
echo 1/3 Embedding index
python src\retrieve.py --build >> %LOG% 2>&1
echo [%time%] 2/3 retrieval eval >> %LOG%
echo 2/3 Retrieval evaluation
python src\eval_retrieval.py >> %LOG% 2>&1
echo [%time%] 3/3 answers >> %LOG%
echo 3/3 Answers with qwen2.5:7b (33 questions)
python src\eval_answers.py >> %LOG% 2>&1
echo [%time%] DONE >> %LOG%
echo Done. Results in the results folder.
pause
