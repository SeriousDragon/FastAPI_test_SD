# Бэкенд: FastAPI + GPU

Этот сервис принимает тексты и изображения, обрабатывает их на GPU и возвращает предсказания.

## Перед стартом
1. Убедитесь, что PyTorch установлен с поддержкой CUDA — следуйте `../README_GPU.md`.
2. Создайте и активируйте отдельное окружение Python 3.10+.
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Запуск
1. Поднимите сервер:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
2. При первом запуске CatBoost обучится на `app/data/train_channels.csv` (обучение происходит на GPU, поэтому первый старт занимает больше времени).
3. После прогрева доступны endpoints:
   - Проверка статуса: `GET http://127.0.0.1:8000/health`
   - Классификация текста (POST JSON):
     ```bash
     curl -X POST "http://127.0.0.1:8000/predict/text" \
          -H "Content-Type: application/json" \
          -d '{"text": "Просто пример текста"}'
     ```
   - Классификация изображения (POST multipart):
     ```bash
     curl -X POST "http://127.0.0.1:8000/predict/image" \
          -F "file=@path/to/image.jpg"
     ```

## Артефакты
- Модель CatBoost и список классов автоматически сохраняются в `app/models/artifacts/`.
- Если удалить файлы артефактов, сервис переобучит модель при следующем запуске.
