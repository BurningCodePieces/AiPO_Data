import easyocr
from googletrans import Translator
import cv2
import urllib.request
from bs4 import BeautifulSoup
import requests
import geocoder
import numpy as np
from collections import Counter
import tkinter as tk
from tkinter import filedialog
import torch

root = tk.Tk()
root.title("Projekt z AiPO")
root.geometry("900x600")

vid = None

reader = easyocr.Reader(['en'])
translator = Translator()

haar_url = 'https://raw.githubusercontent.com/BurningCodePieces/AiPO_Data/main/haar.xml'
haar_path = 'haar.xml'
urllib.request.urlretrieve(haar_url, haar_path)
nPlateCascade = cv2.CascadeClassifier("haar.xml")


def get_text_from_image(img):
    extracted = reader.readtext(img, detail=0)
    return extracted


def get_language_from_text(text):
    try:
        return translator.detect(' '.join(text)).lang
    except:
        return None


def license_plate_from_image(img):
    plates = nPlateCascade.detectMultiScale(img, 1.1)
    results = []
    for plate in plates:
        (x, y, w, h) = plate
        plate_img = img[y:y + h, x:x + w]
        license_plate = reader.readtext(plate_img, detail=0, allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ')
        results.append(license_plate)
    return results


def get_nationality(plate_number):
    cookies = dict(cookies_are='working')
    r = requests.get(
        f'https://www.ofesauto.es/en/know-the-nationality-of-a-vehicle-through-its-plate-number/?matricula={plate_number}',
        cookies=cookies)
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('tbody', class_='resultados-table')
    output = []
    if table:
        zona = table.findAll('div', class_='cell-zona')
        progress = table.findAll('div', class_='progress')
        for j, k in enumerate(zona):
            prob = progress[j].text
            output.append([prob.replace('\n', ''), k.text])
        return output


def check_city(city, min_accuracy=0.2):
    output = []
    g = geocoder.osm(city, maxRows=5)
    for result in g:
        r = result.json
        if r['accuracy'] > min_accuracy:
            if 'country' in r.keys():
                output.append([r['accuracy'], r['address'], r['country']])
            else:
                output.append([r['accuracy'], r['address']])
    return output


def detect_is_on_right_side(img):
    imc = cv2.Canny(img, 100, 200)

    lines = cv2.HoughLinesP(imc, 1, np.pi / 180, 50, None, 50, 5)
    if lines is None:
        return None

    sum_left = 0
    sum_right = 0
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        if y1 < y2:
            if x1 < x2:
                sum_left += 1
            else:
                sum_right += 1
        else:
            if x1 < x2:
                sum_right += 1
            else:
                sum_left += 1
    return sum_right >= sum_left


def get_info_from_frames_at_the_end(output):
    global vid

    if vid is None:
        return

    output.configure(text='Preparing...')
    output.update()

    detected_plates = np.array([])
    detected_text = np.array([])
    detected_is_on_right = 0
    detected_languages = np.array([])

    frames_number = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = vid.read()

    width = 600
    ratio = width / frame.shape[0]
    height = int(frame.shape[0] * ratio)
    dim = (width, height)
    i = 0

    while ret:
        i += 1
        if i % 3 == 0:
            output.configure(text=f'{i}/{frames_number}')
            output.update()

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized_gray = cv2.resize(frame_gray, dim, interpolation=cv2.INTER_AREA)

            if detect_is_on_right_side(frame_resized_gray):
                detected_is_on_right += 1

            plates = license_plate_from_image(frame_resized_gray)

            if len(plates) > 0:
                for number_plate in plates:
                    for n_p in number_plate:
                        detected_plates = np.append(detected_plates, n_p)

            output_text = get_text_from_image(frame_resized_gray)
            output_text_lower = [x.lower() for x in output_text]

            if len(output_text_lower) > 0:
                lang = get_language_from_text(output_text_lower)
                if lang:
                    detected_languages = np.append(detected_languages, lang)

                for text in output_text_lower:
                    detected_text = np.append(detected_text, text)

        ret, frame = vid.read()

    counts_str = ""
    counts_plates = Counter(detected_plates)
    for plate, c in counts_plates.items():
        if c > 1:
            nationality = get_nationality(plate)
            if nationality:
                counts_str += 'PLATE: ' + str(plate) + ' - ' + str(nationality[0][0]) + '\n'

    counts_text = Counter(detected_text)
    i = 0
    j = 0
    for text, _ in counts_text.most_common():
        if j == 5 or i == 5:
            break
        if j < 5 and any(c.isdigit() for c in text) and len(text) > 4:
            nationality = get_nationality(text)
            if nationality:
                j += 1
                counts_str += 'PLATE: ' + str(text) + ' - ' + str(nationality[0][0]) + '\n'
        elif i < 5 and not any(c.isdigit() for c in text) and len(text) > 2:
            city_res = check_city(text)
            if city_res:
                i += 1
                counts_str += 'CITY: ' + str(text) + ' - ' + str(city_res[0][0]) + '\n'
    counts_str += 'RIGHT SIDE OF THE ROAD: ' + str(detected_is_on_right / frames_number * 100) + '%\n'
    counts_languages = Counter(detected_languages)
    counts_str += 'LANGUAGE: ' + str(counts_languages.most_common(2)) + '\n'
    return counts_str


def load_video(video_num, output):
    global vid
    output.configure(text='Loading video...')
    output.update()
    if video_num:
        if video_num == 1:
            video_url = 'https://github.com/BurningCodePieces/AiPO_Data/raw/main/video1.mp4'
            vid_path = 'video1.mp4'
        elif video_num == 2:
            video_url = 'https://github.com/BurningCodePieces/AiPO_Data/raw/main/video2.mp4'
            vid_path = 'video2.mp4'
        elif video_num == 3:
            video_url = 'https://github.com/BurningCodePieces/AiPO_Data/raw/main/video3.mp4'
            vid_path = 'video3.mp4'
        else:
            video_url = 'https://download.ifi.uzh.ch/rpg/web/data/E2VID/datasets/driving_gen3/external_videos/back8.mp4'
            vid_path = 'video4.mp4'
        urllib.request.urlretrieve(video_url, vid_path)
    else:
        vid_path = filedialog.askopenfilename(initialdir="/",
                                              title="Select a video",
                                              filetypes=(("Videos",
                                                          "*.mp4"),))
    vid = cv2.VideoCapture(vid_path)
    output.configure(text='Video loaded. You can click execute to analyze the video')


def print_result(output):
    output.configure(font=("Comic Sans", 36))
    counts_str = get_info_from_frames_at_the_end(output)
    output.configure(font=("Comic Sans", 16))
    output.configure(text=counts_str, wraplength=700)


frame = tk.LabelFrame(root, text="Choose a video:", padx=10, pady=10, height=100, width=300)
frame.pack(padx=10, pady=10, side=tk.BOTTOM)
result_frame = tk.Label(root, text="", padx=10, pady=10, bg="grey", fg="white", width=700, height=100)
result_frame.pack(padx=10, pady=10, side=tk.BOTTOM)

button1 = tk.Button(frame, text="Video 1", command=lambda: load_video(1, result_frame))
button2 = tk.Button(frame, text="Video 2", command=lambda: load_video(2, result_frame))
button3 = tk.Button(frame, text="Video 3", command=lambda: load_video(3, result_frame))
button4 = tk.Button(frame, text="Video 4", command=lambda: load_video(4, result_frame))
button5 = tk.Button(frame, text="Browse files", command=lambda: load_video(None, result_frame))
button6 = tk.Button(frame, text="Execute", command=lambda: print_result(result_frame))

button1.config(width=10, height=4)
button2.config(width=10, height=4)
button3.config(width=10, height=4)
button4.config(width=10, height=4)
button5.config(width=20, height=4)
button6.config(width=20, height=4)

button1.grid(row=0, column=0)
button2.grid(row=0, column=1)
button3.grid(row=0, column=2)
button4.grid(row=0, column=3)
button5.grid(row=1, column=0, columnspan=2)
button6.grid(row=1, column=2, columnspan=2)

if __name__ == '__main__':
    root.mainloop()
