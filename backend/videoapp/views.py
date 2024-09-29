# videoapp/views.py

from django.core.files.storage import FileSystemStorage

from django.views.decorators.csrf import csrf_exempt

from django.views import View
from django.shortcuts import render, redirect
from .forms import VideoUploadForm
from .models import VideoFile
import moviepy.editor as mp
import zipfile
import os

from asgiref.sync import sync_to_async
from django.http import JsonResponse, FileResponse

import torch
import whisper
import cv2
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import asyncio
import subprocess  # Для вызова ffmpeg
import time
from ultralytics import YOLO
from moviepy.config import change_settings




# Загружаем модель YOLO
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8s.pt').to('cpu')


# Укажите путь к ImageMagick
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# Путь для сохранения моделей
MODEL_DIR = "./models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

ffmpeg_path = r'C:\vs_code\Hack_clips\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Проверка доступности CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

# Модель ResNet для анализа видео
weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=weights)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Преобразование для обработки кадра моделью ResNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def check_file_exists(file_path):
    """Проверяет, существует ли файл по указанному пути, и возвращает его."""
    if os.path.exists(file_path):
        print(f"Файл найден: {file_path}")
        return file_path  # Возвращаем путь к файлу, если он найден
    else:
        print(f"Файл не найден: {file_path}")
        return None  # Возвращаем None, если файл не найден


# Функция для извлечения аудио
def extract_audio(input_video_path, audio_output_path):
    # Проверка существования входного видео
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    # Создаем директорию для выходного аудиофайла, если она не существует
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)

    # Команда для извлечения аудио с помощью ffmpeg
    command = [
        'ffmpeg', 
        '-i', input_video_path, 
        '-q:a', '0', 
        '-map', 'a', 
        audio_output_path, 
        '-y'  # Перезапись выходного файла, если он существует
    ]

    # Запуск команды
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Проверка успешности выполнения
    if result.returncode != 0:
        raise RuntimeError(f"Error extracting audio from {input_video_path}")
# # Функция для извлечения аудио
# def extract_audio(video_file_path):
#     audio_output_path = os.path.splitext(video_file_path)[0] + '.wav'
    
#     # Проверка существования видеофайла
#     if not os.path.exists(video_file_path):
#         print(f"Файл не найден: {video_file_path}")
#         return
    
#     command = ['ffmpeg', '-i', video_file_path, '-q:a', '0', '-map', 'a', audio_output_path, '-y']
    
#     try:
#         subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
#         print(f"Аудио извлечено и сохранено в: {audio_output_path}")
#     except subprocess.CalledProcessError as e:
#         print(f"Ошибка при извлечении аудио: {e.stderr.decode()}")

# Функция для объединения видео с аудио
def combine_audio_and_video(processed_video_path, audio_input_path, final_output_path, bitrate='5000k'):
    command = ['ffmpeg', '-i', processed_video_path, '-i', audio_input_path, '-c:v', 'libx264', '-b:v', bitrate, '-c:a', 'aac', '-strict', 'experimental', final_output_path, '-y']
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def log_time(start_time, description):
    elapsed_time = time.time() - start_time
    print(f"{description} завершено за {elapsed_time:.2f} секунд.")

def get_frame_score(frame, model):
    try:
        input_tensor = preprocess(frame).to(device)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)
        score = output.abs().sum().item()
        return score
    except Exception as e:
        print(f"Ошибка при обработке кадра: {e}")
        return 0

def continue_to_end_of_video_dynamics(frame_scores, end_time, max_duration):
    last_dynamic_time = end_time
    for time_stamp, score in frame_scores:
        if time_stamp >= end_time and time_stamp - end_time <= max_duration and score > 0.5:
            last_dynamic_time = time_stamp
        elif time_stamp >= end_time:
            break
    return last_dynamic_time

def extract_audio_from_video(video_path, output_audio_path, target_sample_rate=16000):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path)
        video.close()

        audio = AudioSegment.from_file(output_audio_path)
        audio = audio.set_frame_rate(target_sample_rate)
        audio.export(output_audio_path, format="wav")
        print(f"Аудио извлечено и изменено на частоту {target_sample_rate} Гц в файл: {output_audio_path}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"Файл {audio_path} не найден для транскрипции.")
        return

    model_name = "small"
    model = whisper.load_model(model_name, device=device)
    
    result = model.transcribe(audio_path, verbose=True)
    segments = result['segments']
    transcribed_text = ""
    timing_data = []

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()

        # Фильтрация коротких и малозначимых сегментов
        if len(text.split()) > 2:  # Исключаем очень короткие предложения
            transcribed_text += text + " "
            timing_data.append((start_time, end_time, text))
    
    return transcribed_text, timing_data

def load_local_emotion_model(model_dir):
    # Проверка существования директории
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Загружаем модель и токенизатор
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

def analyze_emotions_with_local_model(sentences, model, tokenizer):
    emotions = []
    
    # Установить устройство (CPU или GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Убедиться, что модель на правильном устройстве
    
    # Веса для различных эмоций
    emotion_weights = {
        'joy': 1,
        'sadness': 0.2,
        'anger': 2,
        'surprise': 1.5,
        'fear': 1.5,
        'disgust': 1
    }
    
    # Обработка предложений
    for sentence in sentences:
        # Токенизация входного предложения и перемещение на устройство
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        # Отключить вычисление градиентов для повышения производительности
        with torch.no_grad():
            outputs = model(**inputs)

        # Вычисление вероятностей и получение метки с максимальной вероятностью
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, label_idx = torch.max(probabilities, dim=1)
        label = model.config.id2label[label_idx.item()]
        
        # Расчет эмоционального балла
        emotion_score = emotion_weights.get(label, 0) * confidence.item()
        emotions.append((sentence, label, emotion_score))
    
    return emotions

def group_sentences_by_semantics(sentences, timing_data, similarity_threshold=0.4, max_group_duration=170):
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    grouped_sentences = []
    current_group = []
    current_group_time = 0
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    current_group.append(sentences[0])
    current_group_time = timing_data[0][1] - timing_data[0][0]
    current_embedding = sentence_embeddings[0].unsqueeze(0)

    for i in range(1, len(sentences)):
        new_sentence = sentences[i]
        new_embedding = sentence_embeddings[i].unsqueeze(0)
        similarity_score = util.pytorch_cos_sim(current_embedding, new_embedding).mean().item()
        
        start_time = timing_data[i][0]
        end_time = timing_data[i][1]
        sentence_duration = end_time - start_time

        # Проверяем, завершен ли контекст предложения
        if similarity_score >= similarity_threshold and current_group_time + sentence_duration <= max_group_duration:
            current_group.append(new_sentence)
            current_group_time += sentence_duration
            current_embedding = torch.cat((current_embedding, new_embedding), dim=0).mean(dim=0).unsqueeze(0)
        else:
            # Останавливаем объединение, если предложение завершено по контексту
            grouped_sentences.append((" ".join(current_group), current_group_time))
            current_group = [new_sentence]
            current_group_time = sentence_duration
            current_embedding = new_embedding

    if current_group:
        grouped_sentences.append((" ".join(current_group), current_group_time))
    
    return grouped_sentences

def continue_text_to_end_of_sentence(timing_data, end_time, max_duration):
    last_valid_end_time = end_time
    for segment_start, segment_end, segment_text in timing_data:
        if segment_start >= end_time and segment_end - end_time <= max_duration:
            last_valid_end_time = segment_end
        elif segment_start >= end_time:
            break
    return last_valid_end_time

def adjust_clip_based_on_text_and_dynamics(segments, timing_data, frame_scores, max_duration=170, min_duration=15):
    adjusted_segments = []
    used_times = []

    for score, (start_time, end_time, text) in segments:
        end_time_text = continue_text_to_end_of_sentence(timing_data, end_time, max_duration)
        end_time_video = continue_to_end_of_video_dynamics(frame_scores, end_time, max_duration)
        adjusted_end_time = max(end_time_text, end_time_video)
        
        # Ограничение длительности клипа
        start_time, adjusted_end_time = adjust_clip_length(start_time, adjusted_end_time, timing_data, min_duration, max_duration)

        # Проверка на пересечение с уже добавленными сегментами
        overlap = False
        for used_start, used_end in used_times:
            if start_time < used_end and adjusted_end_time > used_start:
                overlap = True
                break

        if not overlap:
            adjusted_segments.append((score, (start_time, adjusted_end_time, text)))
            used_times.append((start_time, adjusted_end_time))

    return adjusted_segments

def filter_unique_segments(segments, min_duration=15, max_duration=170):
    unique_segments = []
    last_end_time = 0
    all_texts = []  # Список для хранения текста всех уникальных сегментов

    for score, (start, end, text) in segments:
        duration = end - start
        if duration < min_duration or duration > max_duration:
            continue  # Игнорировать слишком длинные или короткие сегменты
        if start >= last_end_time:  # Проверка на отсутствие пересечения
            unique_segments.append((score, (start, end, text)))
            last_end_time = end
            all_texts.append(text)  # Добавляем текст в общий список

    # Сохраняем весь текст в файл
    with open("unique_segments_texts.txt", "w", encoding='utf-8') as text_file:
        for text in all_texts:
            text_file.write(text + "\n")

    return unique_segments

def remove_trailing_static_content(segments, frame_scores, timing_data, video_duration, text_dynamic_threshold=0.5, video_dynamic_threshold=0.2):
    """Удаляет статичные фрагменты (например, титры) в конце видео."""
    adjusted_segments = []

    for score, (start_time, end_time, text) in segments:
        # Проверяем, является ли это концом видео
        if end_time >= video_duration - 1:
            # Проверяем наличие динамики текста и видео в последних секундах
            end_time_text = continue_text_to_end_of_sentence(timing_data, end_time, max_duration=video_duration - start_time)
            end_time_video = continue_to_end_of_video_dynamics(frame_scores, end_time, video_duration - start_time)
            last_dynamic_time = max(end_time_text, end_time_video)
            
            if last_dynamic_time < end_time and end_time - last_dynamic_time > 3:  # Если последние 3 секунды статичны
                print(f"Удаляем последние {end_time - last_dynamic_time:.2f} секунды статичного контента.")
                end_time = last_dynamic_time  # Убираем статичный конец (например, титры)
        
        adjusted_segments.append((score, (start_time, end_time, text)))

    return adjusted_segments

def adjust_clip_length(start_time, end_time, timing_data, min_duration, max_duration):
    if end_time - start_time < min_duration:
        for segment_start, segment_end, _ in timing_data:
            if segment_start >= end_time:
                new_duration = segment_end - start_time
                if new_duration <= max_duration:
                    end_time = segment_end
                break
    elif end_time - start_time > max_duration:
        last_valid_end_time = start_time + min_duration
        for segment_start, segment_end, _ in timing_data:
            if segment_start >= start_time and segment_start <= end_time:
                if segment_end - start_time <= max_duration:
                    last_valid_end_time = segment_end
                else:
                    break
        end_time = last_valid_end_time
    return start_time, end_time

def evaluate_segments_interest(timing_data, model, tokenizer, frame_scores, text_weight=3.0, video_weight=1.0, group_size=10):
    interest_scores = []
    
    grouped_timing_data = group_sentences_by_semantics(
        [td[2] for td in timing_data], timing_data, max_group_duration=170)
    
    for i, (grouped_text, group_time) in enumerate(grouped_timing_data):
        start_idx = i * group_size
        end_idx = min(len(timing_data) - 1, (i + 1) * group_size - 1)

        if start_idx >= len(timing_data):
            break

        start_time = timing_data[start_idx][0]
        end_time = timing_data[end_idx][1]

        emotions = analyze_emotions_with_local_model([grouped_text], model, tokenizer)
        score = sum([emotion_score for _, _, emotion_score in emotions])

        length_factor = len(grouped_text.split())
        interest_score_text = score * length_factor * text_weight
        
        frame_score = np.mean([fs[1] for fs in frame_scores if start_time <= fs[0] <= end_time])
        interest_score_video = frame_score * video_weight

        total_score = interest_score_text + interest_score_video
        interest_scores.append((total_score, (start_time, end_time, grouped_text)))

    return sorted(interest_scores, key=lambda x: x[0], reverse=True)


def split_subtitle_text(text, max_words=5):
    """
    Разделяет текст субтитров на части по max_words в каждой строке.
    """
    words = text.split()
    split_texts = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return split_texts

def add_subtitles_with_split(
    video_clip, 
    subtitles: list[tuple[float, float, str]], 
    font: str = 'Montserrat-Bold', 
    fontsize: int = 20, 
    color: str = 'white', 
    stroke_color: str = 'black', 
    stroke_width: int = 1, 
    max_words: int = 5, 
    y_offset: float = 0.4
) -> CompositeVideoClip:
    """
    Добавляет субтитры с разделением на части и делает их красивее.
    """
    subtitle_clips = []
    for start, end, text in subtitles:
        split_texts = split_subtitle_text(text, max_words)
        num_splits = len(split_texts)
        split_duration = (end - start) / num_splits

        for i, split_text in enumerate(split_texts):
            txt_clip = (TextClip(
                split_text, 
                fontsize=fontsize, 
                font=font, 
                color=color, 
                stroke_color=stroke_color, 
                stroke_width=stroke_width, 
                method='caption', 
                size=(video_clip.w * 0.8, None), 
                align='center'
            ).set_start(start + i * split_duration)
            .set_end(start + (i + 1) * split_duration)
            .set_position(('center', int(video_clip.h * (1 - y_offset)))))
            
            # Убедитесь, что продолжительность клипа не меньше минимального значения
            if split_duration < 0.1:  # Минимальное время для отображения текста
                txt_clip = txt_clip.set_duration(0.1)

            subtitle_clips.append(txt_clip)

    return CompositeVideoClip([video_clip, *subtitle_clips])


# Вычисление расстояния между объектами
def calculate_distance(obj1, obj2):
    x1, y1, w1, h1 = obj1
    x2, y2, w2, h2 = obj2
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

# Обнаружение основного объекта
def detect_main_object(frame, prev_obj=None):
    small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    results = model(small_frame)

    main_obj = None
    max_conf = 0

    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = result.xyxy[0].cpu().numpy()
        conf = result.conf.cpu().numpy()

        # Увеличиваем координаты обратно в масштаб полного изображения
        current_obj = (int(2 * x_min), int(2 * y_min), int(2 * (x_max - x_min)), int(2 * (y_max - y_min)))
        
        if conf > max_conf:
            max_conf = conf
            main_obj = current_obj

    return main_obj


#  Сглаживание переходов камеры
def smooth_camera_transition(current_coords, target_coords, alpha=0.05):
    x_cur, y_cur, w_cur, h_cur = current_coords
    x_tgt, y_tgt, w_tgt, h_tgt = target_coords

    x_new = int((1 - alpha) * x_cur + alpha * x_tgt)
    y_new = int((1 - alpha) * y_cur + alpha * y_tgt)
    w_new = int((1 - alpha) * w_cur + alpha * w_tgt)
    h_new = int((1 - alpha) * h_cur + alpha * h_tgt)

    return (x_new, y_new, w_new, h_new)

# Добавление черных полос для сохранения пропорций
def add_black_bars(frame, target_width, target_height):
    h, w, _ = frame.shape
    aspect_ratio_frame = w / h
    aspect_ratio_target = target_width / target_height

    if aspect_ratio_frame > aspect_ratio_target:
        new_width = target_width
        new_height = int(target_width / aspect_ratio_frame)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        top_bottom_padding = (target_height - new_height) // 2
        padded_frame = cv2.copyMakeBorder(resized_frame, top_bottom_padding, top_bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    elif aspect_ratio_frame < aspect_ratio_target:
        new_height = target_height
        new_width = int(target_height * aspect_ratio_frame)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        left_right_padding = (target_width - new_width) // 2
        padded_frame = cv2.copyMakeBorder(resized_frame, 0, 0, left_right_padding, left_right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    else:
        padded_frame = cv2.resize(frame, (target_width, target_height))

    return padded_frame

# Проверка на значительное движение объекта
def has_object_moved_enough(new_coords, prev_coords, threshold=100):
    if prev_coords is None:
        return True  # Если нет предыдущих координат, считаем, что объект сместился
    distance = calculate_distance(new_coords, prev_coords)
    return distance > threshold

def crop_and_center(frame, obj_coords, target_coords, target_width, target_height, alpha=0.05):
    """
    Центрирует и обрезает кадр на объекте с применением сглаживания для плавных переходов.
    """
    # Сглаживаем переход между текущими и новыми координатами объекта
    x, y, w, h = smooth_camera_transition(obj_coords, target_coords, alpha)
    x_center = x + w // 2
    y_center = y + h // 2

    # Определяем размеры для кадрирования
    crop_width = int(target_height * (9 / 16))
    crop_height = target_height

    x_crop_start = max(0, min(x_center - crop_width // 2, frame.shape[1] - crop_width))
    y_crop_start = max(0, min(y_center - crop_height // 2, frame.shape[0] - crop_height))

    # Обрезаем кадр вокруг объекта
    cropped_frame = frame[y_crop_start:y_crop_start + crop_height, x_crop_start:x_crop_start + crop_width]
    return cropped_frame

async def process_video_with_centering_and_subtitles(video_path, output_clip_path, subtitles, max_words=3, alpha=0.05, detection_interval=20):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_width = int(target_height * (9 / 16))  # Вертикальный формат 9:16

    # Настройка для сохранения видеоклипа
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    temp_video_path = output_clip_path.replace('.mp4', '_video.mp4')  # Временный файл для видео
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))

    frame_count = 0
    obj_coords = None
    target_coords = None
    prev_coords = None  # Последние известные координаты объекта

    success, frame = cap.read()
    while success:
        if frame_count % detection_interval == 0 or obj_coords is None:
            # Определяем новый объект
            new_target_coords = detect_main_object(frame, prev_coords)
            if new_target_coords is not None:
                if has_object_moved_enough(new_target_coords, prev_coords, threshold=50):
                    target_coords = new_target_coords
                    prev_coords = target_coords
            else:
                target_coords = prev_coords if prev_coords is not None else (frame.shape[1]//2, frame.shape[0]//2, target_width, target_height)

        if target_coords is not None:
            obj_coords = smooth_camera_transition(obj_coords if obj_coords else target_coords, target_coords, alpha)

        # Центрируем и обрезаем кадр
        centered_frame = crop_and_center(frame, obj_coords, target_coords, target_width, target_height, alpha)
        out.write(centered_frame)

        success, frame = cap.read()
        frame_count += 1

    cap.release()
    out.release()

    # Извлекаем аудио из оригинального видео
    audio_output_path = output_clip_path.replace('.mp4', '_audio.aac')
    extract_audio(video_path, audio_output_path)

    # Добавляем красивые субтитры с разделением текста на части
    with VideoFileClip(temp_video_path) as video_clip:
        final_clip = add_subtitles_with_split(video_clip, subtitles, max_words=max_words, font='Montserrat', fontsize=22, color='white', stroke_color='black', stroke_width=1, y_offset=0.4)
        final_clip.write_videofile(output_clip_path.replace(".mp4", "_final.mp4"), codec="libx264", audio_codec="aac")

    # Объединяем видео с аудио
    final_output_path = output_clip_path.replace(".mp4", "_final_with_audio.mp4")
    combine_audio_and_video(output_clip_path.replace(".mp4", "_final.mp4"), audio_output_path, final_output_path)

    # Удаляем временные файлы
    os.remove(temp_video_path)
    os.remove(audio_output_path)

    return final_output_path  # Возвращаем путь к финальному файлу


async def process_video_async(video_path):
    start_time = time.time()
    audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")  # Путь к временному аудиофайлу
    
    # Извлекаем аудио из видео
    extract_audio_from_video(video_path, audio_path)

    # Распознаем аудио и получаем текст
    transcribed_text, timing_data = transcribe_audio(audio_path)

    # Загружаем модель эмоций
    emotion_model_dir = os.path.join(os.path.dirname(__file__), 'models', 'emotion')
    emotion_model, emotion_tokenizer = load_local_emotion_model(emotion_model_dir)

    # Анализируем кадры видео
    cap = cv2.VideoCapture(video_path)
    frame_scores = []
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    success, frame = cap.read()
    frame_idx = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while success:
        frame_idx += 1
        if frame_idx % fps == 0:
            score = get_frame_score(frame, resnet_model)
            current_time = frame_idx / fps
            frame_scores.append((current_time, score))
        success, frame = cap.read()
    cap.release()

    # Оценка интересных сегментов
    top_segments = evaluate_segments_interest(timing_data, emotion_model, emotion_tokenizer, frame_scores, text_weight=3.0, video_weight=1.0)

    # Применяем фильтрацию уникальных сегментов
    final_segments = filter_unique_segments(top_segments)

    # Обрезаем статичные части в конце клипов
    final_segments = remove_trailing_static_content(final_segments, frame_scores, timing_data, video_duration)

    output_clips_path = os.path.join(os.path.dirname(__file__), 'input_video', 'output_clips')

    if not os.path.exists(output_clips_path):
        os.makedirs(output_clips_path)

    # Список для сохраненных путей к видео
    processed_files = []

    # Обрабатываем каждый сегмент с центрированием и добавлением субтитров
    for i, (_, (start, end, _)) in enumerate(final_segments[:10]):
        temp_clip_path = os.path.join(output_clips_path, f"temp_clip_{i + 1}.mp4")

        # Извлекаем под-клип
        with VideoFileClip(video_path) as video:
            subclip = video.subclip(start, end)
            subclip.write_videofile(temp_clip_path, codec="libx264", audio_codec="aac")

        # Центрируем и добавляем субтитры
        clip_subtitles = [
            (s_start - start, s_end - start, text) for (s_start, s_end, text) in timing_data if s_start >= start and s_end <= end
        ]

        # Обрабатываем видео с центрированием и добавлением субтитров, получая финальный файл
        final_clip_path = await process_video_with_centering_and_subtitles(temp_clip_path, temp_clip_path, clip_subtitles, max_words=3)

        # Добавляем путь к обработанному видео в список
        processed_files.append(final_clip_path)  # Добавляем финальный файл

    # Удаляем временный аудиофайл
    if os.path.exists(audio_path):
        os.remove(audio_path)

    log_time(start_time, "Полная асинхронная обработка видео завершена")

    # Возвращаем список обработанных файлов
    return processed_files

# Путь к директории input_video
input_video_dir = os.path.join(os.path.dirname(__file__), 'input_video')
# Создание директории, если она не существует
if not os.path.exists(input_video_dir):
    os.makedirs(input_video_dir)



@csrf_exempt
def upload_and_process_video(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video')

        if video_file:
            video_file_path = os.path.join('temp', video_file.name)

            # Сохраняем файл на диск
            with open(video_file_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            processed_files = asyncio.run(process_video_async(video_file_path))

            print("Созданные файлы:")
            for processed_file in processed_files:
                print(processed_file)  # Вывод созданных файлов

            # Создание zip-архива
            zip_path = os.path.join('temp', 'output.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Добавляем обработанные файлы
                for processed_file in processed_files:
                    if os.path.exists(processed_file):
                        zipf.write(processed_file, os.path.basename(processed_file))
                        print(f"Добавлен файл в архив: {processed_file}")  # Лог для проверки
                    else:
                        print(f"Файл не найден для добавления в архив: {processed_file}")

                # Поиск файлов, которые заканчиваются на _centered_final_with_audio_final_with_audio
                output_clips_path = os.path.join(os.path.dirname(video_file_path), 'input_video', 'output_clips')
                for root, dirs, files in os.walk(output_clips_path):
                    for file in files:
                        if file.endswith('_centered_final_with_audio.mp4'):
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):  # Проверка на существование файла
                                zipf.write(file_path, os.path.basename(file_path))
                                print(f"Добавлен дублирующий файл в архив: {file_path}")  # Лог для проверки

            return FileResponse(open(zip_path, 'rb'), as_attachment=True)

        return JsonResponse({'error': 'Файл не найден или произошла ошибка обработки.'}, status=400)

    return JsonResponse({'error': 'Неподдерживаемый метод.'}, status=405)



def download_file(request, filename):
    file_path = os.path.join('temp', filename)  # Путь к файлу
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True)
    return JsonResponse({'error': 'Файл не найден.'}, status=404)


class VideoUploadView(View):
    def post(self, request, *args, **kwargs):
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.save()  # Сохраняем загруженное видео

            # Получаем путь к видео
            video_path = video_file.file.path  # Предполагается, что ваше поле видео называется "file"

            # Запускаем асинхронную обработку видео
            asyncio.run(process_video_async(video_path))

            return JsonResponse({"status": "success", "video_id": video_file.id})

        return JsonResponse({"status": "error", "errors": form.errors})
    

def video_result(request, video_id):
    video_file = VideoFile.objects.get(id=video_id)
    return render(request, 'videoapp/result.html', {'video_file': video_file})
