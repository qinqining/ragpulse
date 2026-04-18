#!/usr/bin/env python3
"""单独测试 Trafilatura 网页清洗效果：原数据 vs 清洗后"""
import sys
import trafilatura

url = sys.argv[1] if len(sys.argv) > 1 else 'https://example.com'

print(f'>>> 抓取: {url}')
downloaded = trafilatura.fetch_url(url)
if not downloaded:
    print('❌ 抓取失败，请检查 URL 或网络')
    sys.exit(1)

print(f'\n{"=" * 70}')
print('【原始 HTML / 原数据】')
print(f'长度: {len(downloaded)} 字符')
print(f'{"=" * 70}')
print(downloaded[:3000])  # 最多显示3000字符
if len(downloaded) > 3000:
    print(f'\n... (还有 {len(downloaded) - 3000} 字符省略)')

print(f'\n{"=" * 70}')
print('【清洗后文本 / 提取结果】')
print(f'{"=" * 70}')
text = trafilatura.extract(downloaded)
if text:
    print(text)
else:
    print('(无文本内容)')
print(f'{"=" * 70}')
print(f'清洗后长度: {len(text) if text else 0} 字符')

# also show metadata
meta = trafilatura.extract_metadata(downloaded)
if meta:
    print(f'\n--- 页面元信息 ---')
    if meta.title:
        print(f'标题: {meta.title}')
    if meta.author:
        print(f'作者: {meta.author}')
    if meta.date:
        print(f'日期: {meta.date}')
    if meta.description:
        print(f'描述: {meta.description}')
