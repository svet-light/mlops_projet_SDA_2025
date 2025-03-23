FROM python:3.9-slim

#Working Directory
WORKDIR /app

#Install packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

#Copy source code to working directory
COPY .  /app/

EXPOSE 5000

CMD ["python", "app.py"]