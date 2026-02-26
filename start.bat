@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Option-scan — запуск веб-интерфейса (Windows)
echo ============================================================

:: Проверяем наличие виртуального окружения
if not exist ".venv\Scripts\python.exe" (
    echo [ОШИБКА] Виртуальное окружение не найдено.
    echo Сначала запустите install.bat
    pause
    exit /b 1
)

echo Запуск Option-scan (Deribit Options Scanner)...
echo Откройте браузер по адресу: http://localhost:8501
echo Для остановки нажмите Ctrl+C
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
