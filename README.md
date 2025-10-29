# tree-health-infer-service (CPU)

FastAPI сервис для инференса пайплайна: растения → глубина → дефекты → линковка → эвристики → правила → визуализация/экспорт.

## Запуск
```bash
copy .env.example .env
docker compose -f docker-compose.cpu.yml up --build -d
curl -s http://localhost:8000/health
```

## Эндпоинты
- GET /health
- POST /infer  (multipart form, поле `file`)
- POST /debug/depth
- POST /debug/heuristics
- POST /debug/export-smoke

Артефакты пишутся в `./data/out/<request_id>/`.
