FROM valohai/pypermill
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install seaborn numpy pandas matplotlib valohai-utils statsmodels 
RUN pip install scikit-learn==0.24.2


CMD ["python"]
