from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
import os

# Загружаем переменные окружения (в том числе OPENAI_API_KEY)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PlanState(TypedDict):
    profile: str
    ideas: List[str]


def generate_ideas(state: PlanState) -> PlanState:
    """
    Узел графа: генерирует идеи постов с помощью GPT.
    Если GPT недоступен (ошибка/лимит), используется простая заглушка.
    """
    profile = state["profile"]

    prompt = (
        "Ты помогаешь автору вести Telegram-канал.\n"
        f"Профиль канала: {profile}\n"
        "Сгенерируй 5 КОНКРЕТНЫХ идей постов для этого канала.\n"
        "- Пиши по одной идее в строке.\n"
        "- Не добавляй вступления и заключения, только сами идеи.\n"
    )

    ideas: List[str] = []

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",  
            messages=[
                {
                    "role": "system",
                    "content": "Ты помощник по контент-маркетингу для Telegram-каналов."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
        )

        text = response.choices[0].message.content or ""

        # Разбиваем ответ на строки и чистим маркеры
        raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
        ideas = [line.lstrip("-•0123456789. ").strip() for line in raw_lines]

    except Exception as e:
        # На всякий случай выводим ошибку в консоль, чтобы ты видел, если что-то не так
        print("GPT error in generate_ideas:", repr(e))

    # Если GPT не вернул идей — используем fallback-заглушку
    if not ideas:
        short = profile if len(profile) < 80 else profile[:80] + "..."
        ideas = [
            f"Пост-знакомство: расскажи, о чём канал и для кого он: {short}",
            f"Список 5 советов по теме канала: {short}",
            f"Личная история, связанная с темой канала: {short}",
            f"Разбор типичной ошибки подписчиков по теме: {short}",
            f"Подведение итогов недели по теме канала и выводы: {short}",
        ]

    return {"profile": profile, "ideas": ideas}


# Обычный граф LangGraph с одним узлом
graph = StateGraph(PlanState)
graph.add_node("generate_ideas", generate_ideas)
graph.set_entry_point("generate_ideas")
graph.add_edge("generate_ideas", END)

plan_graph = graph.compile()