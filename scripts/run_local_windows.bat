@echo off
rem Builds the embedding index and runs both evaluations with Ollama on this machine.
rem Needs Python 3.10+ and Ollama installed. Takes roughly 30-60 min on a laptop CPU.
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONUTF8=1
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -q -r requirements.txt
ollama pull bge-m3
ollama pull qwen2.5:7b
echo.
echo === 1/3 Embedding index (bge-m3) ===
python src\retrieve.py --build
echo.
echo === 2/3 Retrieval evaluation (bm25 / dense / hybrid) ===
python src\eval_retrieval.py
echo.
echo === 3/3 Answers with qwen2.5:7b (33 questions) ===
python src\eval_answers.py
echo.
echo Done. Results in the results folder.
pause
