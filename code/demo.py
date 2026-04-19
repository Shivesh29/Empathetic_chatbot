import torch
import os
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoTokenizer
from taco_pipeline import TacoModel

EMOTIONS_LIST = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise"
]

emotion_map = {
    "admiration":"happy","amusement":"happy","joy":"happy","love":"happy","excitement":"happy","optimism":"happy","gratitude":"happy","approval":"happy",
    "anger":"angry","annoyance":"angry","disgust":"angry","disapproval":"angry",
    "fear":"fear","nervousness":"fear",
    "sadness":"sad","grief":"sad","disappointment":"sad","remorse":"sad","embarrassment":"sad",
    "confusion":"neutral","curiosity":"neutral","realization":"neutral",
    "surprise":"neutral","desire":"neutral","caring":"neutral","pride":"neutral","relief":"neutral"
}

def run_generative_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model_path = "taco_final.pth"
    print("📂 Looking for model:", model_path)
    print("📁 Files here:", os.listdir())

    if not os.path.exists(model_path):
        print("❌ ERROR: Model file not found. Check filename again.")
        return

    print("✅ Model file found\n")

    
    taco_name = "microsoft/deberta-v3-small"
    t_tokenizer = AutoTokenizer.from_pretrained(taco_name, use_fast=False)
    t_model = TacoModel(taco_name)

    print("📦 Loading classifier weights...")
    t_model.load_state_dict(torch.load(model_path, map_location=device))
    t_model.to(device).eval()

    
    gen_name = "facebook/blenderbot-400M-distill"
    b_tokenizer = BlenderbotTokenizer.from_pretrained(gen_name)
    b_model = BlenderbotForConditionalGeneration.from_pretrained(gen_name).to(device)

    print("\n✨ Chatbot Ready (BlenderBot). Type 'q' to exit.\n")

    while True:
        user_text = input("User: ")
        if user_text.lower() == 'q':
            break

        with torch.no_grad():
            enc = t_tokenizer(user_text, return_tensors="pt", truncation=True, padding=True).to(device)
            u_emb = t_model(enc['input_ids'], enc['attention_mask'])

            l_enc = t_tokenizer(list(EMOTIONS_LIST), padding=True, return_tensors="pt").to(device)
            l_embs = t_model(l_enc['input_ids'], l_enc['attention_mask'])

            scores = torch.matmul(u_emb, l_embs.T)
            raw_emotion = EMOTIONS_LIST[torch.argmax(scores, dim=1).item()]

        emotion = emotion_map.get(raw_emotion.lower(), "neutral")

        
        prompt = f"{user_text} I feel {emotion}."

        inputs = b_tokenizer(prompt, return_tensors="pt").to(device)

        reply_ids = b_model.generate(
            **inputs,
            max_length=80,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            no_repeat_ngram_size=3
        )

        bot_reply = b_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        print(f">> Detected Emotion: {raw_emotion.upper()} → {emotion.upper()}")
        print(f"Bot: {bot_reply}\n")


if __name__ == "__main__":
    run_generative_demo()
