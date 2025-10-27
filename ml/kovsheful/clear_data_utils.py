import os
import re
from pathlib import Path
from typing import Any
import csv
import pandas as pd

def clean_media_headers():
    russian_media = [
        "Российская газета",
        "Московский комсомолец",
        "Новая газета",
        "Комсомольская правда",
        "Независимая газета",
        "Радио Свобода",
        "Первый канал",
        "Globallookpress.com",
        "ТАСС",
        "AP",
        "Reuters",
        "Коммерсантъ»",
        "Фото (c)AFP",
        "Россия 24",
        "ТВ Дождь",
        "The Moscow Times",
        "AFP",
        "РИА Новости",
        "РИА",
        "Новости",
        "РИА-Новости",
        "Global Look",
        "Global Look",
        "Лента.ру»",
        "Wikimedia"
    ]

    usa_media = [
        "The New York Times",
        "The Washington Post",
        "The Wall Street Journal",
        "Los Angeles Times",
        "The Associated Press",
        "NBC News",
        "ABC News",
        "CBS News",
        "Fox News",
        "USA Today",
        "The Boston Globe",
        "Chicago Tribune",
        "San Francisco Chronicle",
        "Miami Herald",
        "New York Daily News",
        "The Dallas Morning News"
    ]

    europe_media = [
        "Financial Times",
        "The Guardian",
        "The Times",
        "The Daily Telegraph",
        "BBC News",
        "Sky News",
        "Le Monde",
        "Le Figaro",
        "France 24",
        "Der Spiegel",
        "Süddeutsche Zeitung",
        "Frankfurter Allgemeine Zeitung",
        "El País",
        "El Mundo",
        "Corriere della Sera",
        "La Repubblica",
        "De Volkskrant",
        "NRC Handelsblad",
        "Dagens Nyheter",
        "Svenska Dagbladet",
        "Helsingin Sanomat",
        "The Irish Times",
        "Gazeta Wyborcza",
        "Hospodářské noviny"
    ]

    all_media_list = russian_media + usa_media + europe_media
    all_media = sorted(set(all_media_list), key=len, reverse=True)
    escaped_media = [re.escape(m) for m in all_media]
    media_pattern = "(?:" + "|".join(escaped_media) + ")"

    pattern_step1 = re.compile(rf"(?:[Фф]ото:\s*.*?|/\s*){media_pattern}")
    pattern_step2_a = re.compile(r"Фото:\s*", re.IGNORECASE)
    pattern_step2_b = re.compile(r"Фото\s*с", re.IGNORECASE)
    leading_separators = re.compile(r'^\s*[:;,\-—–/|]+\s*')

    def clean_text(text: str) -> str:
        s = (text or "").replace("\u00A0", " ")
        while True:
            m = pattern_step1.search(s)
            if not m:
                break
            s = s[m.end():]
        s = pattern_step2_a.sub("", s)
        s = pattern_step2_b.sub("", s)
        if "Фото " in s:
            s = s.replace("Фото ", "")
        changed = True
        while changed:
            changed = False
            for media in all_media:
                if s.startswith(media):
                    s = s[len(media):].lstrip()
                    s = leading_separators.sub("", s)
                    changed = True
                    break
        s = leading_separators.sub("", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def process_file(input_path: str, output_path: str) -> None:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, open(output_path, "w",
                                                                                   encoding="utf-8") as fout:
            for line in fin:
                cleaned = clean_text(line.rstrip("\n"))
                fout.write(cleaned + "\n")

    base = os.path.dirname(__file__)
    input_path = os.path.abspath(os.path.join(base, "../../work_in_progress/lenta_texts_headless.csv"))
    output_path = os.path.abspath(os.path.join(base, "../../work_in_progress/lenta_texts_2000-20.csv"))
    process_file(input_path, output_path)


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def clean_taiga():
    # Matches one or more leading VK mention fragments like "[id123|Имя Фамилия]"
    # with at least one Cyrillic character before the closing bracket.
    _RUSSIAN_CHARS = "\u0400-\u04FF"
    _VK_PREFIX = re.compile(
        rf'^(?:\[id\d+\|[^\]]*?[{_RUSSIAN_CHARS}][^\]]*?\][\s,;:\u2014\u2013-]*)+'
    )

    def _strip_leading_mentions(text: str) -> str:
        """Remove leading VK mentions when the text starts with [id...|Имя]."""
        if not text:
            return ""
        if not text.startswith("[id"):
            return text
        match = _VK_PREFIX.match(text)
        if not match:
            return text
        remainder = text[match.end():]
        return remainder.lstrip(" ,;:\u2014\u2013-")

    def clean_text(value: Any) -> str:
        """Convert value to string and strip VK mentions and extra spaces."""
        if pd.isna(value):
            return ""
        text = str(value).replace("\u00A0", " ")
        text = _strip_leading_mentions(text)
        return text.strip()

    base_dir = Path(__file__).resolve().parent
    input_path = (base_dir / "../../work_in_progress/taiga_style_dataset.csv").resolve()
    output_path = (base_dir / "../../work_in_progress/taiga_style_dataset_clean.csv").resolve()

    df = pd.read_csv(input_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'text' and 'label' in input CSV.")

    df["new_text"] = df["text"].apply(clean_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df[["new_text", "label"]].to_csv(output_path, index=False)


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def clean_photo_headers():
    QUOTES = '"«»“”'

    def process_line(line: str) -> str:
        s = line.strip().strip(QUOTES)
        if '/' not in s and s.startswith('Фото: '):
            return s[6:]
        if '/' not in s:
            return s
        if not s.startswith('Фото: '):
            return s
        idx_slash = s.rfind('/')
        if idx_slash == -1 or idx_slash > 200:
            return s
        temp_substr = s[idx_slash + 1:]
        if s[:3] == "РИА":
            idx_last_char = temp_substr.find('и')
            if idx_last_char == -1:
                return temp_substr
            return temp_substr[idx_last_char + 2:]
        else:
            idx_first_space = temp_substr.find(' ')
            return temp_substr[idx_first_space + 1:]

    def delete_ria_header(line: str) -> str:
        s = line.strip().strip(QUOTES)
        if s.startswith("РИА Новости "):
            return s[12:]
        else:
            return s

    def process_files(inputs, out_path):
        with open(out_path, 'w', encoding='utf-8') as fout:
            for path in inputs:
                with open(path, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        s = process_line(line)
                        s = delete_ria_header(s)
                        fout.write(s + '\n')

    in_paths = ["work_in_progress/lenta_texts.csv"]
    out_path = "work_in_progress/lenta_texts_headless.csv"
    process_files(in_paths, out_path)


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def extract_rbc_text_to_csv():
    def normalize(s: str) -> str:
        return re.sub(r'\s+', ' ', s).strip()

    def is_blocked(s: str, prefixes: tuple[str, ...], limit: int = 400) -> bool:
        s = s.lstrip()
        return (len(s) < limit and s.casefold().startswith(prefixes)) or len(s) < limit * 0.5

    STOP_WORDS = ["Наталья Щербакова Департамент аналитической информации РБК",
                  "Департамент аналитической информации",
                  "Валентина Гаврикова Департамент аналитической информации РБК",
                  'РБК',
                  "Quote.rbc.ru РБК-Недвижимость: ",
                  "РБК-Недвижимость: ",
                  '"РБК. Личные финансы":',
                  ]
    STOP_PREFIXES = tuple(w.casefold() for w in STOP_WORDS)

    excel_path = "../../input_data/rbc_2000_2020.xlsx"
    output_csv = "../../work_in_progress/rbc_texts.csv"

    df = pd.read_excel(excel_path, engine="openpyxl", dtype=str)
    texts = df["text"].dropna().astype(str)

    seen = set()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for value in texts:
            v = normalize(value)
            if is_blocked(v, STOP_PREFIXES, limit=400) and v in seen:
                continue
            writer.writerow([v])
            seen.add(v)


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def extract_xlsx_text_to_csv():
    def normalize(s: str) -> str:
        return re.sub(r'\s+', ' ', s).strip()

    excel_path = "../../input_data/lentaRu_2000_2020.xlsx"
    output_csv = "../../work_in_progress/lenta_texts.csv"

    df = pd.read_excel(excel_path, engine="openpyxl", dtype=str)
    texts = df["text"].dropna().astype(str)

    seen = set()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for value in texts:
            v = normalize(value)
            if v in seen or len(v) < 200:
                continue
            writer.writerow([v])
            seen.add(v)


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def split_csv_chunks():
    def write_chunk(rows, start_idx, end_idx, base_name):
        if not rows:
            return
        name = f"splitted_csv/{base_name}_{start_idx}-{end_idx}.csv"
        with open(name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)

    input_path = "clean_data/lenta_texts.csv"
    chunk_size = 100
    base_name = "lenta_texts"
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        buffer = []
        start = 1
        idx = 0
        for row in reader:
            idx += 1
            buffer.append(row)
            if len(buffer) == chunk_size:
                end = start + len(buffer) - 1
                write_chunk(buffer, start, end, base_name)
                start = end + 1
                buffer = []
        if buffer:
            end = start + len(buffer) - 1
            write_chunk(buffer, start, end, base_name)


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def trim_after_marker():
    def trim_field(value, marker):
        if value is None:
            return value
        s = str(value)
        idx = s.find(marker)
        if idx == -1:
            return s
        return s[idx + len(marker):]

    input_path = "headerless_all.csv"
    output_path = "../../clean_data/lenta_data.csv"
    marker = "$$$###"
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", newline="",
                                                              encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            new_row = [trim_field(cell, marker) for cell in row]
            writer.writerow(new_row)