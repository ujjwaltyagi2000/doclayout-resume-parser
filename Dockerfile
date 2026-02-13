FROM public.ecr.aws/lambda/python:3.9

# Install OpenJDK
RUN yum install -y java-1.8.0-openjdk


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["app.handler"]
