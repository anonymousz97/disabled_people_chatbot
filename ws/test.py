import time

from browser_func import Search


def main():
    search_tool = Search()
    search_tool.search("Ai là triệu phú")
    for i in range(10):
        search_tool.scroll_down()
    for i in range(10):
        search_tool.scroll_up()
    search_tool.navigate(3)
    search_tool.back()
    search_tool.navigate(0)

if __name__ == '__main__':
    main()
