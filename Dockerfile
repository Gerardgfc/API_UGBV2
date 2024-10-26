# Usar una imagen base de Python
FROM python:3.11

# Establecer el directorio de trabajo
WORKDIR /main

# Copiar los archivos necesarios al contenedor
COPY . .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que se ejecutará la API
EXPOSE 5000

# Comando para ejecutar la API
#CMD ["gunicorn", "-b", "0.0.0.0:5000", "main:app"]
CMD ["gunicorn", "-b", "0.0.0.0:" + str(os.environ.get('PORT', 5000)), "main:app"]
