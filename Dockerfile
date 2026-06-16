FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /workspace

# README와 동일: Python 3.11 conda 환경
RUN conda create -y -n xai_demo python=3.11 \
    && conda clean -afy

ENV PATH=/opt/conda/envs/xai_demo/bin:$PATH

# Blender EEVEE 렌더: NVIDIA OpenGL/EGL + Xvfb 가상 디스플레이 (docker run 시 --gpus all 필요)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV DISPLAY=:99
ENV XVFB_DISPLAY=:99
ENV XVFB_RESOLUTION=1920x1080x24

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxrender1 libxext6 libxi6 libxfixes3 libxxf86vm1 \
    libxkbcommon0 libxkbcommon-x11-0 \
    libgl1 libglu1-mesa libsm6 libice6 \
    libegl1 libglvnd0 libopengl0 libgl1-mesa-dri \
    xvfb x11-utils \
    libfontconfig1 libfreetype6 libjpeg-turbo8 libpng16-16 \
    libgomp1 libtbb12 libglib2.0-0 libdbus-1-3 libxcb1 wget \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.3.1 + CUDA 12.1
RUN pip install --no-cache-dir torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.blender.org/pypi/ \
    -r /tmp/requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# Windows CRLF 방지: shebang 대신 bash로 직접 실행 + 줄바꿈 정규화
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/bin/bash", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
