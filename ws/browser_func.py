import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import validators

class Search:
    """
    Class for search and web surfing
    ...

    Attributes
    ----------
    result_list = hyperlink for all result
    current_content = current search url

    Methods
    -------
    search(content = str): Search google by content word
    E.g: search("Ai là triệu phú")
    navigate(index = int): Go to the link with index th
    back(): back to previous page
    scroll_up()
    scroll_dowm()

    Example in test.py file
    """

    def __init__(self):
        self.results_list = list
        self.current_content = ""
        options = Options()
        options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=options)
        self.set_window_to_right_half()

    def set_window_to_right_half(self):
        # Get the screen dimensions
        screen_width = self.driver.execute_script("return window.screen.availWidth")
        screen_height = self.driver.execute_script("return window.screen.availHeight")

        # Set browser window size to half the screen width and full height
        half_screen_width = screen_width // 2
        self.driver.set_window_size(half_screen_width, screen_height)

        # Set browser window position to the right half of the screen
        self.driver.set_window_position(half_screen_width, 0)

    def search(self, content: str):
        self.driver.get(url="https://www.google.com.vn/")
        self.driver.implicitly_wait(2.0)
        search = self.driver.find_element("name", 'q')
        search.send_keys(content)
        search.send_keys(Keys.RETURN)
        self.driver.implicitly_wait(4.0)
        self.get_result_list()
        self.driver.implicitly_wait(2.0)
        self.current_content = self.driver.current_url

    def get_result_list(self):
        results_list = self.driver.find_elements(By.TAG_NAME, 'cite')

        for i in range(len(results_list)):
            results_list[i] = results_list[i].text.replace(">", "/").replace("›", "/").replace(" ", "")
            if not validators.url(results_list[i]):
                results_list[i] = ''

        self.results_list = list(filter(None, results_list))
        return self.results_list

    def navigate(self, index: int):
        self.driver.get(self.results_list[index-1])
        self.driver.implicitly_wait(2.0)

    def back(self):
        previous = self.current_content if self.current_content != "" else "https://www.google.com.vn/"
        self.driver.get(previous)
        self.driver.implicitly_wait(2.0)

    def scroll_down(self):
        self.driver.execute_script("scrollBy(0," + str(200) + ")")
        time.sleep(0.5)

    def scroll_up(self):
        self.driver.execute_script("scrollBy(0," + str(-200) + ")")
        time.sleep(0.5)


# # Example usage:
# search = Search()
# search.search("Ai là triệu phú")
# time.sleep(10)  # Keep the browser open for 10 seconds to verify the position
# search.driver.quit()