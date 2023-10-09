#!/usr/bin/env python3
# -*- coding utf-8 -*-

from playwright.sync_api import sync_playwright

'''
playwright调用演示
pip install playwright
playwright install
'''

def run():
    # 使用Playwright上下文管理器
    with sync_playwright() as p:
        # 使用Chromium，但你也可以选择firefox或webkit
        browser = p.chromium.launch()
        
        # 创建一个新的页面
        page = browser.new_page()
        
        # 导航到指定的URL
        page.goto('https://langchain.com/')
        
        # 获取并打印页面标题
        title = page.title()
        print(f"Page title is: {title}")
        
        # 关闭浏览器
        browser.close()

if __name__ == "__main__":
    run()