import torch
import os
from turkish_news_classifier import TurkishNewsClassifier

def test_with_manual_texts():
    """Test the model with manually input texts"""
    
    # Check if trained model exists
    model_path = 'turkish_news_model.pth'
    vocab_path = 'vocabulary.pkl'
    
    if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
        print("❌ Trained model not found!")
        print("You need to train the model first by running:")
        print("python turkish_news_classifier.py")
        print("\nOr make sure the following files exist:")
        print(f"- {model_path}")
        print(f"- {vocab_path}")
        return
    
    # Load the trained model
    print("🔄 Loading trained model...")
    classifier = TurkishNewsClassifier()
    
    try:
        classifier.load_model(model_path, vocab_path)
        print("✅ Model loaded successfully!")
        print(f"📊 Model can classify into {len(classifier.label_encoder.classes_)} categories:")
        for i, category in enumerate(classifier.label_encoder.classes_):
            print(f"   {i+1}. {category}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # Interactive testing loop
    print("🧪 Manual Text Testing")
    print("=" * 50)
    print("Enter Turkish news texts to classify (type 'quit' to exit)")
    print("Examples:")
    print("- Türkiye ekonomisi büyümeye devam ediyor")
    print("- Futbol maçında gol atıldı")
    print("- Yeni teknoloji geliştirildi")
    print("- Sağlık bakanlığı açıklama yaptı")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            text = input("\n📝 Enter Turkish text: ").strip()
            
            # Check for exit command
            if text.lower() in ['quit', 'exit', 'çık', 'çıkış']:
                print("👋 Goodbye!")
                break
            
            # Check if text is empty
            if not text:
                print("⚠️  Please enter some text!")
                continue
            
            # Make prediction
            print("🔄 Processing...")
            predicted_category, confidence = classifier.predict(text)
            
            # Display results
            print(f"📊 Results:")
            print(f"   Text: '{text}'")
            print(f"   Predicted Category: {predicted_category}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # Add confidence interpretation
            if confidence > 0.8:
                print("   🎯 High confidence prediction")
            elif confidence > 0.6:
                print("   ✅ Good confidence prediction")
            elif confidence > 0.4:
                print("   ⚠️  Low confidence prediction")
            else:
                print("   ❓ Very low confidence - result may be unreliable")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")


def test_with_predefined_examples():
    """Test the model with predefined example texts"""
    
    # Check if trained model exists
    model_path = 'turkish_news_model.pth'
    vocab_path = 'vocabulary.pkl'
    
    if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
        print("❌ Trained model not found!")
        print("Please train the model first by running: python turkish_news_classifier.py")
        return
    
    # Load the trained model
    print("🔄 Loading trained model...")
    classifier = TurkishNewsClassifier()
    classifier.load_model(model_path, vocab_path)
    print("✅ Model loaded successfully!\n")
    
    # Predefined example texts for each category
    test_examples = {
        "siyaset": [
            "Başbakan dün parlamentoda önemli açıklamalar yaptı",
            "Seçim sonuçları açıklandı ve yeni hükümet kurulacak",
            "Anayasa değişikliği önerisi mecliste görüşülüyor"
        ],
        "ekonomi": [
            "Türkiye ekonomisi bu yıl yüzde 5 büyüdü",
            "Dolar kuru düşüş eğiliminde devam ediyor",
            "Enflasyon oranları geçen aya göre azaldı"
        ],
        "spor": [
            "Galatasaray dün akşam önemli bir galibiyet aldı",
            "Türk milli takımı dünya kupasına hazırlanıyor",
            "Olimpiyat oyunlarında altın madalya kazanıldı"
        ],
        "teknoloji": [
            "Yeni yapay zeka sistemi geliştirildi",
            "Akıllı telefon teknolojisinde devrim niteliğinde gelişme",
            "Türk mühendisler uzay teknolojisinde ilerleme kaydetti"
        ],
        "saglik": [
            "Sağlık bakanlığı yeni tedavi yöntemini onayladı",
            "Korona virüs aşısı kitle uygulaması başladı",
            "Kalp hastalıkları için yeni ameliyat tekniği bulundu"
        ],
        "dunya": [
            "Amerika Birleşik Devletleri yeni politika açıkladı",
            "Avrupa Birliği ülkeleri önemli karar aldı",
            "Dünya liderleri iklim değişikliği konusunda anlaştı"
        ],
        "kultur": [
            "İstanbul film festivali başladı",
            "Türk müziği dünya sahnesinde büyük ilgi gördü",
            "Sanat müzesinde yeni sergi açıldı"
        ]
    }
    
    print("🧪 Testing with Predefined Examples")
    print("=" * 60)
    
    total_tests = 0
    correct_predictions = 0
    
    for expected_category, texts in test_examples.items():
        print(f"\n📂 Testing {expected_category.upper()} category:")
        print("-" * 40)
        
        for text in texts:
            predicted_category, confidence = classifier.predict(text)
            total_tests += 1
            
            # Check if prediction is correct
            is_correct = predicted_category == expected_category
            if is_correct:
                correct_predictions += 1
            
            # Display result
            status = "✅" if is_correct else "❌"
            print(f"{status} Text: '{text}'")
            print(f"   Expected: {expected_category} | Predicted: {predicted_category}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print()
    
    # Summary
    accuracy = (correct_predictions / total_tests) * 100
    print("=" * 60)
    print(f"📊 SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Correct Predictions: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print("=" * 60)


def main():
    """Main function to choose testing mode"""
    print("🏷️  Turkish News Classification - Model Testing")
    print("=" * 50)
    print("Choose testing mode:")
    print("1. Manual text input (interactive)")
    print("2. Test with predefined examples")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                test_with_manual_texts()
                break
            elif choice == '2':
                test_with_predefined_examples()
                break
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("⚠️  Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main() 