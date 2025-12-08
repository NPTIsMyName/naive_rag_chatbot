import os
import json
from tqdm import tqdm
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# ==== CONFIG ====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INPUT_FILE = "vnexpress_kinhdoanh.json"
OUTPUT_FILE = "summaries.json"

# ==== INIT MODEL ====
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)

# ==== HÀM TÓM TẮT ====
def summarize_article(article):
    title = article.get("title")
    content = article.get("content")
    url = article.get("url")

    if not content or not title:
        return {"summary": None, "error": "Missing content/title"}

    prompt = (
        "You are a professional financial news summarizer. "
        "Summarize the following Vietnamese news article briefly "
        "while keeping it informative and factually accurate. "
        "Do not hallucinate or add extra details. "
        "Preserve all factual data such as dates, numbers, and names exactly as they appear. "
        "Respond in Vietnamese.\n\n"
        f"Tiêu đề: {title}\n\n"
        f"Nội dung: {content}\n\n"
        "Tóm tắt:"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        return {"summary": summary, "error": None}
    except Exception as e:
        return {"summary": None, "error": str(e)}

# ==== MAIN ====
if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for article in tqdm(data, desc="Summarizing articles"):
        result = summarize_article(article)
        article["summary"] = result["summary"]
        article["error"] = result["error"]
        results.append(article)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Summaries saved to {OUTPUT_FILE}")
