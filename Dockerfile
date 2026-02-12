FROM public.ecr.aws/lambda/python:3.10

# 1️⃣ Install required system libraries for Docling / PyMuPDF
RUN yum install -y \
    gcc \
    gcc-c++ \
    make \
    mesa-libGL \
    glib2 \
    tar \
    gzip \
    && yum clean all

# 2️⃣ Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# 3️⃣ Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 5️⃣ Copy application code
COPY . .

# 6️⃣ Lambda handler
CMD ["docling_groq.handler"]
