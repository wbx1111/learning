# 指定基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 环境变量：关闭Python缓冲+中文编码+路径
ENV PYTHONUNBUFFERED=1
ENV LANG=zh_CN.UTF-8
ENV LC_ALL=zh_CN.UTF-8
ENV PATH="/usr/local/bin:$PATH"

# 提前创建data目录并赋予权限
RUN mkdir -p /app/data && chmod 777 /app/data

# 安装系统依赖（不修改apt源，直接用默认源+精简依赖）
RUN set -eux; \
    # 安装基础工具和编译依赖（Debian 12适配）
    apt-get update; \
    apt-get install -y --no-install-recommends \
        gcc \
        libxml2-dev \
        libxslt1-dev \
        ca-certificates \
        curl \
        gnupg \
        ; \
    # 清理apt缓存，减小镜像体积
    rm -rf /var/lib/apt/lists/*; \
    # 确保pip存在并升级
    python -m ensurepip --upgrade; \
    # 配置pip国内源（重点：只改pip源，避开apt源的坑）
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    # 升级pip和基础工具
    pip install --no-cache-dir --upgrade pip setuptools wheel;

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 启动命令
CMD ["python", "spider_main.py"]
