import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import torchaudio  # Keep if needed, but not used now
import torch
import pydub
from pydub import AudioSegment
import nltk
import subprocess
import tempfile
import shutil
from TTS.api import TTS

nltk.download('punkt', quiet=True)

class AudiobookGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audiobook Generator")
        self.setGeometry(100, 100, 400, 200)
        
        layout = QVBoxLayout()
        
        self.label = QLabel("Select an EPUB file")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        self.select_button = QPushButton("Select EPUB")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)
        
        self.generate_button = QPushButton("Generate Audiobook")
        self.generate_button.clicked.connect(self.generate_audiobook)
        self.generate_button.setEnabled(False)
        layout.addWidget(self.generate_button)
        
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        layout.addWidget(self.progress)
        
        self.setLayout(layout)
        
        self.epub_path = None
        self.tts = None
        self.reference_wav = "johnny_cash_reference.wav"
        
        if not os.path.exists(self.reference_wav):
            QMessageBox.warning(self, "Warning", "Place 'johnny_cash_reference.wav' in the app directory for Johnny Cash voice.")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select EPUB", "", "EPUB files (*.epub)")
        if file_path:
            self.epub_path = file_path
            self.label.setText(os.path.basename(file_path))
            self.generate_button.setEnabled(True)

    def generate_audiobook(self):
        if not self.epub_path:
            return
        
        temp_dir = None
        try:
            self.progress.setValue(0)
            self.generate_button.setEnabled(False)
            
            # Load TTS model
            device = "cpu"  # Use CPU to avoid crashes; change to "mps" if stable
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            
            # Extract text and chapters
            book = epub.read_epub(self.epub_path)
            title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "Audiobook"
            chapters = []
            chapter_titles = []
            doc_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            total_docs = len(doc_items)
            
            for i, item in enumerate(doc_items):
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text().strip()
                if text:
                    chapters.append(text)
                    ch_title = soup.title.string if soup.title else f"Chapter {i+1}"
                    chapter_titles.append(ch_title)
                self.progress.setValue(int(10 + (i / total_docs) * 10))  # 10-20% for extraction
            
            if not chapters:
                raise ValueError("No text found in EPUB")
            
            # Generate audio segments
            audio_segments = []
            current_time = 0
            chapter_metadata = []
            total_chapters = len(chapters)
            temp_dir = tempfile.mkdtemp()
            
            for idx, chapter_text in enumerate(chapters):
                sentences = nltk.sent_tokenize(chapter_text)
                chapter_audio = AudioSegment.silent(duration=0)
                
                for sent_idx, sent in enumerate(sentences):
                    if not sent.strip():
                        continue
                    wav_path = os.path.join(temp_dir, f"temp_{idx}_{sent_idx}.wav")
                    self.tts.tts_to_file(text=sent, speaker_wav=self.reference_wav, language="en", file_path=wav_path)
                    sent_audio = AudioSegment.from_wav(wav_path)
                    chapter_audio += sent_audio
                    os.remove(wav_path)
                
                if len(chapter_audio) > 0:
                    audio_segments.append(chapter_audio)
                    chapter_metadata.append({
                        'start': current_time,
                        'end': current_time + len(chapter_audio),
                        'title': chapter_titles[idx]
                    })
                    current_time += len(chapter_audio)
                
                self.progress.setValue(int(20 + (idx / total_chapters) * 70))  # 20-90% for TTS
            
            # Concatenate audios
            full_audio = AudioSegment.silent(duration=0)
            for seg in audio_segments:
                full_audio += seg
            
            m4a_path = title.replace(" ", "_") + ".m4a"
            full_audio.export(m4a_path, format="ipod")  # m4a format
            
            # Use ffmpeg to add chapters and metadata
            m4b_path = title.replace(" ", "_") + ".m4b"
            metadata_file = os.path.join(temp_dir, "metadata.txt")
            with open(metadata_file, "w") as f:
                f.write(";FFMETADATA1\n")
                f.write(f"title={title}\n")
                f.write(f"artist=Johnny Cash Voice\n")
                f.write(f"album={title}\n")
                current_ms = 0
                for ch in chapter_metadata:
                    f.write("[CHAPTER]\n")
                    f.write("TIMEBASE=1/1000\n")
                    f.write(f"START={current_ms}\n")
                    f.write(f"END={current_ms + (ch['end'] - ch['start'])}\n")
                    f.write(f"title={ch['title']}\n")
                    current_ms += (ch['end'] - ch['start'])
            
            subprocess.run([
                "ffmpeg", "-i", m4a_path, "-i", metadata_file, 
                "-map_metadata", "1", "-codec", "copy", m4b_path
            ], check=True)
            
            os.remove(m4a_path)
            os.remove(metadata_file)
            shutil.rmtree(temp_dir)
            
            self.progress.setValue(100)
            QMessageBox.information(self, "Success", f"Audiobook generated: {m4b_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.generate_button.setEnabled(True)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudiobookGenerator()
    window.show()
    sys.exit(app.exec_())
