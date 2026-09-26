# 1. Partir de una imagen base que ya tiene Python 3.12
FROM python:3.12-slim

# 2. Definir la carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar primero solo el requirements.txt
COPY requirements.txt .

# 4. Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código
COPY . .

# 6. Documentar que la app usa el puerto 8000
EXPOSE 8000

# 7. El comando que arranca la app cuando el contenedor corre
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
