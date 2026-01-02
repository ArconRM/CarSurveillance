#!/bin/bash

echo "Сборка основного образа modelsinference..."
docker build -f models-inference/Dockerfile -t modelsinference:latest ./models-inference --no-cache

if [ $? -eq 0 ]; then
    echo "Основной образ успешно собран: modelsinference"
else
    echo "Ошибка при сборке основного образа"
    exit 1
fi
