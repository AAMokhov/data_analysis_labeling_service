#!/usr/bin/env python3
"""
Главная точка входа для сервиса анализа и маркировки данных
"""

import os
import sys
import logging
from pathlib import Path

# Добавление текущей директории и директории app в путь Python
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "app"))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Главная функция для запуска приложения"""
    try:
        logger.info("Запуск сервиса анализа и маркировки данных")

        # Импорт и запуск Dash приложения
        from app.dash_app import app

        logger.info("Dash приложение успешно инициализировано")
        logger.info("Запуск веб-сервера на http://0.0.0.0:8050")

        # Запуск приложения
        app.run(
            debug=True,  # Установить в False для продакшена
            host='0.0.0.0',
            port=8050,
            threaded=True,
			dev_tools_ui=True
        )

    except Exception as e:
        logger.error(f"Ошибка запуска приложения: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
