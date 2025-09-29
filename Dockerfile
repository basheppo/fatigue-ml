FROM valohai/pypermill
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip

CMD ["python"]
