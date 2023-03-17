import os
import gdown

outpout_dir = "data"
url = "https://drive.google.com/file/d/1wtogytIpZC74Tq6s1GWaPWY3VBhmOrJy/view?usp=sharing"
output = "data/spam_emails_1.csv"

def download_data():
    if not os.path.exists(output):
        os.makedirs(outpout_dir, exist_ok=True)
        gdown.download(url, output, quiet=False , fuzzy=True)
