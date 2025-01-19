# CS331_VietnameseImageCaptioning
Step 1: 
+ Download CLIPCap model "best_model.pt" at: https://drive.google.com/drive/folders/1SwFyuba28ImHkL5dC7BP8nJ0JfTho55R?usp=sharing
+ and EfficientNetV2+Transformer model "BestModel.pth" & Vocabulary "vocabulary_data.pkl" at: https://drive.google.com/drive/folders/147T8H_m3gSzu1eGyaj9Bu7IN0ZsRnxTi?usp=sharing

Step 2: 
+ Save "best_model.pt" at: ./models/CLIPCap
+  Save "BestModel.pth" and "vocabulary_data.pkl" at: ./models/EfficientNetV2_Transformer

Step 3: Open the terminal at CS431_DemoImageCaptioning

Step 4: Run the command pip install -r requirements.txt

Step 5: Run the command streamlit run App.py
