import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

import urllib
import time

class_index = 14
url = "https://www.google.com/search?q=%D9%83%D9%88%D8%B1%D9%86%D9%8A%D8%B4+%D8%AC%D8%AF%D8%A9&sca_esv=5199b8d8fdf4ca4e&hl=en&biw=1504&bih=668&udm=2&sxsrf=AHTn8zpfLs4weJF2kBcGA9jGHmf5e8wP9A%3A1743477807431&ei=L1zrZ4uLGsGjkdUPlP2m-QE&ved=0ahUKEwjL3or78LWMAxXBUaQEHZS-KR8Q4dUDCBE&uact=5&oq=%D9%83%D9%88%D8%B1%D9%86%D9%8A%D8%B4+%D8%AC%D8%AF%D8%A9&gs_lp=EgNpbWciE9mD2YjYsdmG2YrYtCDYrNiv2KkyBRAAGIAEMgUQABiABDIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHki0DlCyCFiyCHADeACQAQCYAbgBoAG4AaoBAzAuMbgBA8gBAPgBAvgBAZgCBKAC1gHCAgYQABgHGB6YAwCIBgGSBwMzLjGgB4YE&sclient=img"
# Configure WebDriver
driver_path = "chromedriver-linux64/chromedriver"  # Replace with the actual path to the downloaded driver
service = Service(driver_path)
# Launch Browser and Open the URL
counter = 0
numOfPics = 1000
driver = uc.Chrome(service=service)

# Create url variable containing the webpage for a Google image search.
driver.get(url)

input("Press Enter to proceed")
img_results = driver.find_elements(By.XPATH, "//img[contains(@class, 'YQ4gaf')]")

image_urls = []
for img in img_results:
    image_urls.append(img.get_attribute('src'))

folder_path = f'locations/{class_index}/' # change your destination path here

for i in range(numOfPics):
    counter += 1
    try:
        urllib.request.urlretrieve(str(image_urls[i]), f"{folder_path}{counter}")
    except Exception as e:
       print(e)

driver.quit()


        
