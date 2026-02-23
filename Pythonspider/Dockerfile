# 指定基础镜像（保持不变，但确保拉取最新版）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 环境变量：关闭Python缓冲+指定pip路径（解决pip找不到的问题）
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:$PATH"

# 先修复apt源（解决slim镜像源可能失效的问题），再安装依赖
RUN set -eux; \
    # 替换debian源为国内镜像（解决网络问题）
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources; \
    # 更新apt并安装依赖（添加必要的基础工具）
    apt-get update; \
    apt-get install -y --no-install-recommends \
        gcc \
        libxml2-dev \
        libxslt1-dev \
        # 补充基础工具，避免命令找不到
        ca-certificates \
        curl \
        ; \
    # 清理apt缓存
    rm -rf /var/lib/apt/lists/*; \
    # 确认pip存在并升级pip
    python -m ensurepip --upgrade; \
    # 配置pip国内源（清华源）
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    # 安装项目依赖
    pip install --no-cache-dir --upgrade pip setuptools wheel;

# 复制依赖文件（单独拆分，利用Docker缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 启动命令
CMD ["python", "spider_main.py"]