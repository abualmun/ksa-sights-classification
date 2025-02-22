import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

import urllib
import time

class_index = 4
url = "https://www.google.com/search?q=%D8%A8%D8%B1%D8%AC+%D8%A7%D9%84%D9%85%D9%85%D9%84%D9%83%D8%A9&sca_esv=ec4e1a3f479b83b9&udm=2&biw=1368&bih=710&sxsrf=AHTn8zpP8IhVG_5o-XU2Ies1OrC_etAbAQ%3A1739274824249&ei=SDqrZ9vfDuW3i-gPzJ2ImAE&oq=%D8%A7%D9%84%D9%85%D9%85%D9%84%D9%83%D8%A9&gs_lp=EgNpbWciDtin2YTZhdmF2YTZg9ipKgIIADIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeMgYQABgHGB4yBhAAGAcYHjIGEAAYBxgeSMkTUNAEWOQOcAF4AJABAJgBnQSgAfsFqgEHMi0xLjUtMbgBAcgBAPgBAZgCA6ACggbCAgUQABiABJgDAIgGAZIHCTEuMC4xLjAuMaAH1Ag&sclient=img"


# Configure WebDriver
driver_path = "chromedriver-linux64/chromedriver"  # Replace with the actual path to the downloaded driver
service = Service(driver_path)
# Launch Browser and Open the URL
counter = 0
numOfPics = 500
driver = uc.Chrome(service=service)

# Create url variable containing the webpage for a Google image search.
driver.get(url)

input("Press Enter to proceed")
img_results = driver.find_elements(By.XPATH, "//img[contains(@class, 'YQ4gaf')]")

image_urls = []
for img in img_results:
    image_urls.append(img.get_attribute('src'))

folder_path = f'output/{class_index}/' # change your destination path here

for i in range(numOfPics):
    counter += 1
    try:
        urllib.request.urlretrieve(str(image_urls[i]), f"{folder_path}{counter}")
    except Exception as e:
       print(e)

driver.quit()


        
