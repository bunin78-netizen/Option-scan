@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Option-scan — установка зависимостей (Windows)
echo ============================================================

:: Проверяем наличие Python (python или py launcher)
set "PY_CMD="
python --version >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
if not defined PY_CMD (
    py -3 --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3"
)
if not defined PY_CMD (
    echo [ОШИБКА] Python не найден. Установите Python 3.8+ с https://python.org
    pause
    exit /b 1
)

:: Создаём виртуальное окружение, если его нет
if not exist ".venv\Scripts\python.exe" (
    echo Создание виртуального окружения...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать виртуальное окружение.
        pause
        exit /b 1
    )
)

:: Устанавливаем зависимости через python из venv
set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ОШИБКА] Python в виртуальном окружении не найден.
    pause
    exit /b 1
)

echo Установка зависимостей...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ОШИБКА] Не удалось обновить pip.
    pause
    exit /b 1
)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ОШИБКА] Не удалось установить зависимости.
    pause
    exit /b 1
)

:: Создаём .env если его нет
if not exist ".env" (
    if not exist ".env.example" (
        echo [ПРЕДУПРЕЖДЕНИЕ] Файл .env.example не найден. Пропуск создания .env
    ) else (
        echo Создание файла .env из примера...
        copy .env.example .env >nul
        if errorlevel 1 (
            echo [ПРЕДУПРЕЖДЕНИЕ] Не удалось создать файл .env
        )
    )
)

echo.
echo ============================================================
echo  Установка завершена! Запустите start.bat для старта.
echo ============================================================
pause
