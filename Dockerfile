FROM public.ecr.aws/lambda/python:3.10

# 1. Install essential system libraries
RUN yum install -y \
    gcc \
    gcc-c++ \
    make \
    mesa-libGL \
    glib2 \
    tar \
    gzip \
    && yum clean all

# 2. Copy requirements
COPY requirements.txt .

# 3. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install Torch CPU first (the heaviest part)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyMuPDF forcing the binary (prefer-binary flag is key here)
RUN pip install --no-cache-dir --prefer-binary PyMuPDF==1.24.10

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy code
COPY . .

CMD [ "section_extractor.handler" ]