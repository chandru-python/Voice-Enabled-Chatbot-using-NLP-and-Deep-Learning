# 🎤 AI Voice Companion - Voice Enabled Chatbot using NLP & Deep Learning

AI Voice Companion is an intelligent conversational assistant that supports both text and voice interaction. It uses Natural Language Processing (NLP) and a Deep Learning model to understand user intent and generate meaningful responses in real time.

## 🚀 Features
- 🎙️ Voice + Text Input Support  
- 🧠 NLP Processing (Tokenization, Lemmatization)  
- 📊 Bag-of-Words Vectorization  
- 🤖 Deep Learning Intent Classification  
- 🔊 Text-to-Speech Response  
- 🖥️ GUI Interface using Tkinter  
- 📈 Training Graphs (Accuracy & Loss)  
- 📁 Model Saving & Testing  


## ⚙️ Installation
git clone https://github.com/chandru-python/Voice-Enabled-Chatbot-using-NLP-and-Deep-Learning.git  
cd Voice-Enabled-Chatbot-using-NLP-and-Deep-Learning  
pip install -r requirements.txt  

## 🧠 Training the Model
Run:
python src/train_chatbot.py  

Outputs generated:
- Model: model/chatbot_model.h5  
- Logs: outputs/training_output.txt  
- Accuracy graph: outputs/accuracy.png  
- Loss graph: outputs/loss.png  

## 📈 Training Graphs
Accuracy graph shows how model performance improves during training.  
Loss graph shows how error decreases over time.  
All graphs are saved in the outputs folder.

## 📊 Model Performance
Final Training Accuracy: (update with your value, e.g., 92%)  
Loss decreases significantly across epochs and model learns intent patterns effectively.

## 🧪 Testing the Chatbot
Run:
python src/test_chatbot.py  

Supports:
- Text input  
- Voice input (if enabled)  

## 💻 Run GUI
python test_chatbot.py  

## 🔄 Working Flow
User Input (Text/Voice) → Speech to Text → NLP Processing → Bag-of-Words → Deep Learning Model → Intent Prediction → Response → Text + Voice Output

## 🧠 Technologies Used
Python, TensorFlow, Keras, NLTK, NumPy, Scikit-learn, SpeechRecognition, gTTS, Tkinter

## ⚠️ Limitations
- No context awareness (Bag-of-Words limitation)  
- Limited dataset  
- No conversation memory  

## 🚀 Future Improvements
- Use BERT / Transformers  
- Add context awareness  
- Expand dataset  
- Deploy as web/mobile app  
- Add multilingual support  

## 📌 Conclusion
This project demonstrates how NLP, Deep Learning, and Speech Technologies can be integrated to build an intelligent chatbot system capable of natural human-computer interaction.

## 👨‍💻 Author
Chandru  
GitHub: https://github.com/chandru-python
