from text_treatment import TextDetector, TextClassifier, TextTranslator
import os

# source /mnt/c/Users/matheus.melotti/Projects/my/repositories/venvs/tradutor_imagem/bin/activate
# python3 /mnt/c/Users/matheus.melotti/Projects/my/repositories/tradutor_imagem/main.py

def main():
    input_image = os.path.join('resources', '2f49720e-0c7d-4e2d-91d9-ad6940bbb20b.jpeg')
    output_json = os.path.join('output', 'resultados.json')
    output_image = os.path.join('output', 'imagem_com_deteccoes.jpeg')
    
    detector = TextDetector()
    detector.process_document(input_image, output_json)
    detector.visualize_detections(input_image, output_image)
    
    detector.text_format(vertical_threshold=200, horizontal_threshold=100)
    
    classifier = TextClassifier()
    print("\nClassificação de idioma por grupo:")
    for i, text in enumerate(detector.formated_text, 1):
        print(f"\nGrupo {i}:")
        language_results = classifier.classify_text(text)
        for lang, confidence in language_results.items():
            print(f"{lang}: {confidence:.2%}")

    print("\nIniciando tradução...")
    translator = TextTranslator()
    translated_text = translator.translate_text_group(detector.formated_text)
    
    print("\nTexto original e tradução:")
    for i, (original, translated) in enumerate(zip(detector.formated_text, translated_text), 1):
        print(f"\nGrupo {i}:")
        print(f"Original: {original}")
        print(f"Tradução: {translated}")

if __name__ == "__main__":
    main() 
