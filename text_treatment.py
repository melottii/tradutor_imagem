import os
import cv2
import json
import torch
import numpy as np
from doctr.io import DocumentFile
from googletrans import Translator
from typing import List, Dict, Union
from doctr.models import ocr_predictor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import asyncio


class TextDetector:
    def __init__(self):
        """
        Inicializa o detector de texto usando doctr.
        """
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        self.results = []
        self.text_groups = []

    def _calculate_distance(self, box1: Dict, box2: Dict) -> tuple:
        """
        Calcula as distâncias entre dois boxes de texto.
        Retorna a distância vertical e se há sobreposição horizontal.
        """
        # Calcula centros verticais
        center1_y = box1['bbox']['y'] + box1['bbox']['height'] / 2
        center2_y = box2['bbox']['y'] + box2['bbox']['height'] / 2
        vertical_distance = abs(center2_y - center1_y)
        
        # Verifica sobreposição horizontal
        box1_left = box1['bbox']['x']
        box1_right = box1['bbox']['x'] + box1['bbox']['width']
        box2_left = box2['bbox']['x']
        box2_right = box2['bbox']['x'] + box2['bbox']['width']
        
        # Calcula a distância horizontal entre os boxes
        if box1_right < box2_left:  # box1 está à esquerda de box2
            horizontal_distance = box2_left - box1_right
        elif box2_right < box1_left:  # box2 está à esquerda de box1
            horizontal_distance = box1_left - box2_right
        else:  # há sobreposição
            horizontal_distance = 0
            
        return vertical_distance, horizontal_distance

    def _group_text_boxes(self, vertical_threshold: float = 50, horizontal_threshold: float = 100) -> List[List[Dict]]:
        """
        Agrupa boxes de texto baseado na proximidade vertical e horizontal.
        
        Args:
            vertical_threshold: Distância máxima vertical para considerar boxes como parte do mesmo grupo
            horizontal_threshold: Distância máxima horizontal para considerar boxes como parte do mesmo grupo
            
        Returns:
            Lista de grupos de texto, onde cada grupo é uma lista de boxes
        """
        if not self.results:
            return []

        # Ordena os boxes por posição Y
        sorted_boxes = sorted(self.results, key=lambda x: x['bbox']['y'])
        
        groups = [[sorted_boxes[0]]]
        
        for box in sorted_boxes[1:]:
            # Verifica a distância com todos os boxes do último grupo
            last_group = groups[-1]
            min_vertical_dist = float('inf')
            min_horizontal_dist = float('inf')
            
            for last_box in last_group:
                vert_dist, horiz_dist = self._calculate_distance(last_box, box)
                min_vertical_dist = min(min_vertical_dist, vert_dist)
                min_horizontal_dist = min(min_horizontal_dist, horiz_dist)
            
            # Se estiver próximo o suficiente vertical e horizontalmente, adiciona ao grupo atual
            if min_vertical_dist <= vertical_threshold and min_horizontal_dist <= horizontal_threshold:
                groups[-1].append(box)
            else:
                # Cria um novo grupo
                groups.append([box])
        
        # Ordena cada grupo por posição X e depois Y
        for group in groups:
            group.sort(key=lambda x: (x['bbox']['y'], x['bbox']['x']))
            
        return groups

    def detect_text(self, input_path: Union[str, List[str]]) -> List[Dict]:
        """
        Detecta texto em uma imagem, PDF ou lista de imagens.
        
        Args:
            input_path: Caminho para o arquivo (imagem/PDF) ou lista de caminhos
            
        Returns:
            Lista de dicionários com texto e coordenadas
        """
        if isinstance(input_path, str):
            if input_path.lower().endswith('.pdf'):
                doc = DocumentFile.from_pdf(input_path)
            else:
                doc = DocumentFile.from_images(input_path)
        else:
            doc = DocumentFile.from_images(input_path)
        
        result = self.model(doc)
        
        text_boxes = []
        
        if isinstance(input_path, str) and not input_path.lower().endswith('.pdf'):
            img = cv2.imread(input_path)
            height, width = img.shape[:2]
        
        for page_idx, page in enumerate(result.pages):
            if isinstance(input_path, list):
                img = cv2.imread(input_path[page_idx])
                height, width = img.shape[:2]
            
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        bbox = word.geometry
                        x_min, y_min = bbox[0]
                        x_max, y_max = bbox[1]
                        
                        x = int(x_min * width)
                        y = int(y_min * height)
                        w = int((x_max - x_min) * width)
                        h = int((y_max - y_min) * height)
                        
                        text_box = {
                            'text': word.value,
                            'bbox': {
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h
                            },
                            'confidence': float(word.confidence),
                            'page': page_idx + 1
                        }
                        text_boxes.append(text_box)
        
        return text_boxes

    def save_results(self, output_path: str):
        """
        Salva os resultados em um arquivo JSON.
        
        Args:
            results: Lista de dicionários com textos e coordenadas
            output_path: Caminho para salvar o arquivo JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def process_document(self, input_path: Union[str, List[str]], output_path: str) -> None:
        """
        Processa um documento (imagem, PDF ou lista de imagens) e salva os resultados.
        
        Args:
            input_path: Caminho do arquivo de entrada ou lista de caminhos
            output_path: Caminho para salvar os resultados
        """
        self.results = self.detect_text(input_path)
        self.save_results(output_path)

    def visualize_detections(self, image_path: str, output_path: str) -> None:
        """
        Cria uma visualização das detecções em uma imagem.
        
        Args:
            image_path: Caminho da imagem de entrada
            output_path: Caminho para salvar a imagem com as detecções
        """
        image = cv2.imread(image_path)
        
        results = self.detect_text(image_path)
        
        for result in results:
            bbox = result['bbox']
            text = result['text']
            conf = result['confidence']
            
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = f"{text} ({conf:.2f})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image) 
        
    def text_format(self, vertical_threshold: float = 50, horizontal_threshold: float = 100):
        """
        Formata o texto detectado em grupos baseados na proximidade espacial.
        
        Args:
            vertical_threshold: Distância máxima vertical para considerar boxes como parte do mesmo grupo
            horizontal_threshold: Distância máxima horizontal para considerar boxes como parte do mesmo grupo
        """
        self.text_groups = self._group_text_boxes(vertical_threshold, horizontal_threshold)
        
        # Formata cada grupo separadamente
        formatted_groups = []
        for group in self.text_groups:
            group_text = " ".join([box['text'] for box in group])
            formatted_groups.append(group_text)
        
        print("\nTextos detectados por grupo:")
        for i, text in enumerate(formatted_groups, 1):
            print(f"Grupo {i}: {text}")
        
        self.formated_text = formatted_groups


class TextClassifier:
    def __init__(self, text=""):
        self.model_ckpt = "papluca/xlm-roberta-base-language-detection"
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt)
        self.tokenized_text = None
        if text:
            self.text_classification = self.classify_text()

    def _clean_text(self, text: str) -> str:
        """
        Limpa o texto removendo caracteres especiais e formatações indesejadas.
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo
        """
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def tokenize(self, text: str = None) -> None:
        if text is not None:
            self.text = text
        if not self.text:
            raise ValueError("No text provided for tokenization")
            
        # Limpa o texto antes de tokenizar
        cleaned_text = self._clean_text(self.text)
        self.tokenized_text = self.tokenizer(cleaned_text, padding=True, truncation=True, return_tensors="pt")

    def classify_text(self, text: str = None):
        if text is not None:
            self.tokenize(text)
        elif self.tokenized_text is None:
            self.tokenize(self.text)
            
        with torch.no_grad():
            outputs = self.model(**self.tokenized_text)
            logits = outputs.logits
            
        preds = torch.softmax(logits, dim=-1)
        id2lang = self.model.config.id2label
        vals, idxs = torch.max(preds, dim=1)
        
        results = {id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)}
        return results


class TextTranslator:
    def __init__(self):
        self.translator = Translator(service_urls=['translate.googleapis.com'])

    async def translate_text_async(self, text: str) -> str:
        try:
            await asyncio.sleep(0.5)
            
            result = await self.translator.translate(text, dest='pt')
            translated = result.text if result else text
            
            if translated.upper() == text.upper():
                print(f"Tradução retornou o mesmo texto: {text}")
                print("Tentando novamente com 'The' no início...")
                
                if not text.upper().startswith('THE '):
                    new_text = f"The {text}"
                    await asyncio.sleep(0.5)
                    result  = await self.translator.translate(new_text, dest='pt')
                    translated = result.text if result else text
                    print(f"Nova tradução: {translated}")
            
            return translated
            
        except Exception as e:
            print(f"Erro ao traduzir texto: {e}")
            return text

    def translate_text(self, text: str) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.translate_text_async(text))
        finally:
            loop.close()

    async def translate_group_async(self, text_group: List[str]) -> List[str]:
        translated_texts = []
        for text in text_group:
            translated = await self.translate_text_async(text)
            translated_texts.append(translated)
            await asyncio.sleep(1)
        return translated_texts

    def translate_text_group(self, text_group: List[str]) -> List[str]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.translate_group_async(text_group))
        finally:
            loop.close()
