@echo off
echo ============================================================
echo  Option-scan — запуск веб-интерфейса (Windows)
echo ============================================================

:: Проверяем наличие виртуального окружения
if not exist ".venv\Scripts\activate.bat" (
    echo [ОШИБКА] Виртуальное окружение не найдено.
    echo Сначала запустите install.bat
    pause
    exit /b 1
)

:: Активируем виртуальное окружение
call .venv\Scripts\activate.bat

echo Запуск Option-scan (Deribit Options Scanner)...
echo Откройте браузер по адресу: http://localhost:8501
echo Для остановки нажмите Ctrl+C
echo.
streamlit run app.py
pause
