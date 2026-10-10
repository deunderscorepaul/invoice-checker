@echo off
REM Convenience wrapper so the commands in the README work without remembering the runtime path.
REM Usage: run.cmd selftest | generate -n 200 | run --baseline | eval --all | lint | bench
setlocal
set PY=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m invoice_checker %*
