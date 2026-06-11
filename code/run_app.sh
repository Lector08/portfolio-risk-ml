#!/bin/bash
# Скрипт запуску інтерфейсу системи управління ризиками
# Використання: ./run_app.sh

echo "================================================"
echo "  Portfolio Risk Manager — запуск інтерфейсу"
echo "================================================"

cd "$(dirname "$0")"

# Перевірка Streamlit
if python3 -m streamlit --version &>/dev/null; then
    echo "✅ Streamlit знайдено"
    echo "🚀 Запуск на http://localhost:8501"
    python3 -m streamlit run app.py \
        --server.port 8501 \
        --server.headless false \
        --browser.gatherUsageStats false
else
    echo "⚠️  Streamlit не знайдено. Встановлення..."
    python3 -m pip install streamlit --quiet
    echo "✅ Streamlit встановлено"
    echo "🚀 Запуск на http://localhost:8501"
    python3 -m streamlit run app.py
fi
