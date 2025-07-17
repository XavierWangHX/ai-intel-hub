# Stage 1: Browser and build tools installation
#FROM python:3.13.3-slim-bookworm AS install-browser
FROM python:3.11.4-slim-bullseye AS install-browser

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak \
    # 清空原始 sources.list 并添加阿里云源 (以 Debian 11 Bullseye 为例)
    && echo "" > /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian/ bullseye main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb-src https://mirrors.aliyun.com/debian/ bullseye main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian/ bullseye-updates main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb-src https://mirrors.aliyun.com/debian/ bullseye-updates main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian-security/ bullseye-security main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb-src https://mirrors.aliyun.com/debian-security/ bullseye-security main contrib non-free" >> /etc/apt/sources.list \
    # 更新 APT 缓存
    && apt-get update

    
# Install Chromium, Chromedriver, Firefox, Geckodriver, and build tools in one layer
RUN apt-get update \
    && apt-get install -y gnupg wget ca-certificates --no-install-recommends \
    && ARCH=$(dpkg --print-architecture) \
    && wget -qO - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=${ARCH}] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y chromium chromium-driver \
    && chromium --version && chromedriver --version \
    && apt-get install -y --no-install-recommends firefox-esr build-essential \
    && GECKO_ARCH=$(case ${ARCH} in amd64) echo "linux64" ;; arm64) echo "linux-aarch64" ;; *) echo "linux64" ;; esac) \
    && wget https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-${GECKO_ARCH}.tar.gz \
    && tar -xvzf geckodriver-v0.36.0-${GECKO_ARCH}.tar.gz \
    && chmod +x geckodriver \
    && mv geckodriver /usr/local/bin/ \
    && rm geckodriver-v0.36.0-${GECKO_ARCH}.tar.gz \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt lists to reduce image size

# Stage 2: Python dependencies installation
FROM install-browser AS seres-researcher-install

ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /usr/src/app

# Copy and install Python dependencies in a single layer to optimize cache usage
COPY ./requirements.txt ./requirements.txt
COPY ./HarmonyOS_Sans_SC /usr/share/fonts

# === Pip 源配置开始 ===
RUN pip install --upgrade pip -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com && \
    pip install  -r requirements.txt --upgrade --prefer-binary -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
# === Pip 源配置结束 ===
# Stage 3: Final stage with non-root user and app
FROM seres-researcher-install AS seres-researcher

# Basic server configuration
ARG HOST=0.0.0.0
ENV HOST=${HOST}
ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Uvicorn parameters used in CMD
ARG WORKERS=16
ENV WORKERS=${WORKERS}

# Create a non-root user for security
# NOTE: Don't use this if you are relying on `_check_pkg` to pip install packages dynamically.
RUN useradd -ms /bin/bash seres-researcher && \
    chown -R seres-researcher:seres-researcher /usr/src/app && \
    # Add these lines to create and set permissions for outputs directory
    mkdir -p /usr/src/app/outputs && \
    chown -R seres-researcher:seres-researcher /usr/src/app/outputs && \
    chmod 777 /usr/src/app/outputs
USER seres-researcher
WORKDIR /usr/src/app

# Copy the rest of the application files with proper ownership
COPY --chown=seres-researcher:seres-researcher ./ ./
CMD uvicorn main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}

