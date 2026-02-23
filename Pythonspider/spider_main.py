import requests
from bs4 import BeautifulSoup
import time
import csv
import os
# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://movie.douban.com/'
}


def get_movie_info(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'

        # 优先使用 lxml，兼容性更好
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except:
            soup = BeautifulSoup(response.text, 'html.parser')

        grid_view = soup.find('ol', class_='grid_view')
        if not grid_view:
            print(f"警告：在 {url} 未找到电影列表容器")
            return []

        movie_list = grid_view.find_all('li')
        movies = []

        for movie in movie_list:
            # 1. 标题
            title_tag = movie.find('span', class_='title')
            title = title_tag.get_text(strip=True) if title_tag else '未知'

            # 2. 评分
            rating_tag = movie.find('span', class_='rating_num')
            rating = rating_tag.get_text(strip=True) if rating_tag else '0.0'

            # 3. 评分人数
            people_tag = movie.find('span', string=lambda s: s and '人评价' in s)
            people = people_tag.get_text(strip=True) if people_tag else '0人评价'

            # 4. 简介 (核心修改点)
            # 使用 CSS 选择器定位 class 为 inq 的 span
            quote_tag = movie.select_one('span.inq')
            if quote_tag:
                quote = quote_tag.get_text(strip=True)
            else:
                # 豆瓣 Top250 中确实有极少数电影没有这行一句话简介
                quote = "（暂无简评）"

            movie_info = {
                '标题': title,
                '评分': rating,
                '评分人数': people,
                '简介': quote
            }
            movies.append(movie_info)
            print(f"成功获取: 《{title}》 | 简介: {quote}")

        return movies

    except Exception as e:
        print(f"爬取页面 {url} 时发生错误: {e}")
        return []


def save_to_csv(data, filename='douban_top250_fixed.csv'):
    # 强制使用容器内的 /app/data 目录
    target_dir = "/app/data"
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, filename)

    if not data:
        print("[警告] 无数据可保存")
        return

    headers = ['标题', '评分', '评分人数', '简介']
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

    print(f"\n[完成] 数据已保存至 {file_path}")


def main():
    base_url = 'https://movie.douban.com/top250?start={}&filter='
    all_movies = []

    for offset in range(0, 250, 25):
        url = base_url.format(offset)
        print(f"\n--- 正在爬取第 {offset // 25 + 1} 页 ---")
        page_movies = get_movie_info(url)
        all_movies.extend(page_movies)

        # 适当延长延时，豆瓣最近封控较严
        time.sleep(3)

    save_to_csv(all_movies)


if __name__ == '__main__':
    main()