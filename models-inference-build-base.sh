#!/bin/bash

echo "Сборка базового образа с зависимостями..."
docker build -f models-inference/Dockerfile.base -t inference-dependencies:latest ./models-inference

if [ $? -eq 0 ]; then
    echo "Базовый образ успешно собран: inference-dependencies:latest"
else
    echo "Ошибка при сборке базового образа"
    exit 1
fi
