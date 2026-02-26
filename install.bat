@echo off
echo ============================================================
echo  Option-scan — установка зависимостей (Windows)
echo ============================================================

:: Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден. Установите Python 3.8+ с https://python.org
    pause
    exit /b 1
)

:: Создаём виртуальное окружение, если его нет
if not exist ".venv" (
    echo Создание виртуального окружения...
    python -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать виртуальное окружение.
        pause
        exit /b 1
    )
)

:: Активируем виртуальное окружение и устанавливаем зависимости
echo Установка зависимостей...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
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
