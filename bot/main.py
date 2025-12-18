import asyncio
import os

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramForbiddenError
from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
)

from asyncio import to_thread
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import select

from bot.graph_plan import plan_graph
from bot.db import init_db, SessionLocal, User, Draft

load_dotenv()  # Загружаем переменные из .env

print("BOT_TOKEN from env:", bool(os.getenv("BOT_TOKEN")))
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Клиент OpenAI только для генерации постов/идей
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Создаём объекты бота и диспетчера
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

# Фабрика сессий к БД (инициализируем в main())
session_factory = None


# ---------- FSM СОСТОЯНИЯ ----------

class PlanForm(StatesGroup):
    profile = State()        # для /idea: описание канала
    own_idea = State()       # для /idea: у пользователя уже есть идея поста
    generated_post = State() # для /idea: сгенерированный ИИ пост по идее пользователя


class DraftForm(StatesGroup):
    confirm = State()     # подтверждение, что пользователь хочет идти по шагам
    idea = State()        # шаг: идея поста
    title = State()       # шаг: заголовок
    body = State()        # шаг: основной текст
    conclusion = State()  # шаг: заключение / призыв


class DeleteDraftForm(StatesGroup):
    waiting_for_id = State()       # ждём номер черновика (по списку пользователя)
    waiting_for_confirm = State()  # ждём подтверждение на удаление


class EditDraftForm(StatesGroup):
    waiting_for_id = State()       # ждём номер черновика (по списку пользователя)
    waiting_for_text = State()     # ждём новый текст черновика


class SendDraftForm(StatesGroup):
    waiting_for_number = State()   # ждём номер черновика
    waiting_for_channel = State()  # ждём @канал или chat_id


class SaveMediaDraftForm(StatesGroup):
    waiting_for_media = State()    # ждём медиа (фото/видео/видеозаметка/док/войс) с подписью


class EditGeneratedPostForm(StatesGroup):
    editing = State()              # режим редактирования сгенерированного поста
    waiting_for_ai_edit = State()  # ожидание запроса на редактирование ИИ
    waiting_for_media = State()    # ожидание медиа для прикрепления


class RewriteForm(StatesGroup):
    waiting_for_text = State()     # ждём текст для рерайта


class HashtagsForm(StatesGroup):
    waiting_for_text = State()     # ждём текст для генерации хештегов


class VariantsForm(StatesGroup):
    waiting_for_text = State()     # ждём текст для генерации вариантов


class ContentPlanForm(StatesGroup):
    waiting_for_topic = State()    # ждём тему канала для контент-плана
    waiting_for_period = State()   # ждём период (неделя/месяц)


class StyleCopyForm(StatesGroup):
    waiting_for_example = State()  # ждём пример поста
    waiting_for_topic = State()    # ждём тему нового поста


class TemplateForm(StatesGroup):
    choosing_template = State()    # выбор шаблона
    filling_template = State()     # заполнение шаблона


class SearchForm(StatesGroup):
    waiting_for_query = State()    # ждём поисковый запрос


# ---------- КОНСТАНТЫ ----------

DRAFTS_PER_PAGE = 5  # черновиков на страницу
MEDIA_PER_PAGE = 3   # медиа-драфтов на страницу


# ---------- КЛАВИАТУРА ----------

main_menu_kb = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="📝 Создать пост"),
            KeyboardButton(text="📂 Черновики"),
        ],
        [
            KeyboardButton(text="🤖 ИИ-инструменты"),
            KeyboardButton(text="📅 Планирование"),
        ],
        [
            KeyboardButton(text="🔍 Поиск"),
            KeyboardButton(text="❓ Помощь"),
        ],
    ],
    resize_keyboard=True,
)


# ---------- HELPER ФУНКЦИИ ----------


async def get_user_id_from_context(message: types.Message, state: FSMContext = None) -> int:
    """
    Получает telegram_id пользователя.
    Сначала проверяет state (для вызовов из callback), потом message.from_user.id.
    """
    if state:
        data = await state.get_data()
        stored_id = data.get("_user_telegram_id")
        if stored_id:
            return stored_id
    
    # Если message.from_user существует и это не бот
    if message.from_user and not message.from_user.is_bot:
        return message.from_user.id
    
    # Fallback — пытаемся получить из chat (для личных чатов chat.id == user.id)
    return message.chat.id


# ---------- ФУНКЦИИ ДЛЯ РАБОТЫ С БД ----------

async def get_or_create_user(telegram_id: int) -> int:
    """
    Возвращает id пользователя в таблице users.
    Если пользователя нет — создаёт.
    """
    async with session_factory() as session:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if user:
            return user.id

        user = User(telegram_id=telegram_id)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.id


async def create_draft(telegram_id: int, idea_text: str, draft_text: str):
    """
    Создаёт черновик для пользователя.
    """
    user_id = await get_or_create_user(telegram_id)

    async with session_factory() as session:
        draft = Draft(user_id=user_id, idea_text=idea_text, draft_text=draft_text)
        session.add(draft)
        await session.commit()


async def get_user_drafts(telegram_id: int, limit: int = 5):
    """
    Возвращает список черновиков пользователя (последние N).
    """
    user_id = await get_or_create_user(telegram_id)

    async with session_factory() as session:
        result = await session.execute(
            select(Draft)
            .where(Draft.user_id == user_id)
            .order_by(Draft.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


async def get_user_drafts_full(telegram_id: int):
    """
    Возвращает ВСЕ черновики пользователя, отсортированные по времени создания (старые -> новые).
    """
    user_id = await get_or_create_user(telegram_id)

    async with session_factory() as session:
        result = await session.execute(
            select(Draft)
            .where(Draft.user_id == user_id)
            .order_by(Draft.created_at.asc())
        )
        return result.scalars().all()


async def get_user_draft_by_id(telegram_id: int, draft_id: int):
    """
    Возвращает один черновик пользователя по его ID или None, если он не принадлежит пользователю.
    """
    user_id = await get_or_create_user(telegram_id)

    async with session_factory() as session:
        result = await session.execute(
            select(Draft).where(Draft.id == draft_id, Draft.user_id == user_id)
        )
        return result.scalar_one_or_none()


async def delete_user_draft(telegram_id: int, draft_id: int) -> bool:
    """
    Удаляет один черновик пользователя по ID.
    Возвращает True, если что‑то удалили, и False, если черновика не было.
    """
    user_id = await get_or_create_user(telegram_id)

    async with session_factory() as session:
        result = await session.execute(
            select(Draft).where(Draft.id == draft_id, Draft.user_id == user_id)
        )
        draft = result.scalar_one_or_none()
        if not draft:
            return False

        await session.delete(draft)
        await session.commit()
        return True


# ---------- ОБРАБОТЧИКИ КОМАНД ----------

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    # Проверяем, новый ли пользователь
    user_id = await get_or_create_user(message.from_user.id)

    welcome_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Начать работу", callback_data="start:begin")],
            [InlineKeyboardButton(text="📖 Как пользоваться?", callback_data="start:tutorial")],
        ]
    )

    await message.answer(
        "<b>👋 Привет! Я ИИ-ассистент для контента.</b>\n\n"
        "Помогу тебе:\n"
        "• Генерировать идеи для постов\n"
        "• Писать тексты с помощью ИИ\n"
        "• Создавать и хранить черновики\n"
        "• Публиковать посты в канал\n"
        "• Составлять контент-планы\n\n"
        "Выбери, с чего начать:",
        reply_markup=welcome_kb,
    )


@dp.callback_query(lambda c: c.data == "start:begin")
async def cb_start_begin(callback: types.CallbackQuery, state: FSMContext):
    """Начать работу — показать главное меню"""
    await callback.message.edit_text(
        "Отлично! Используй кнопки внизу экрана для навигации. 👇"
    )
    await callback.message.answer("Главное меню:", reply_markup=main_menu_kb)
    await callback.answer()


@dp.callback_query(lambda c: c.data == "start:tutorial")
async def cb_start_tutorial(callback: types.CallbackQuery, state: FSMContext):
    """Показать мини-туториал"""
    tutorial_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Далее →", callback_data="tutorial:2")],
        ]
    )

    await callback.message.edit_text(
        "<b>📖 Как пользоваться ботом</b>\n\n"
        "<b>Шаг 1: Создание поста</b>\n\n"
        "Нажми «📝 Создать пост» и выбери:\n"
        "• <b>Сгенерировать идеи</b> — ИИ предложит темы\n"
        "• <b>Написать черновик</b> — пошаговое создание\n"
        "• <b>Сохранить медиа</b> — фото/видео с подписью",
        reply_markup=tutorial_kb,
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "tutorial:2")
async def cb_tutorial_2(callback: types.CallbackQuery, state: FSMContext):
    """Туториал шаг 2"""
    tutorial_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Далее →", callback_data="tutorial:3")],
            [InlineKeyboardButton(text="← Назад", callback_data="start:tutorial")],
        ]
    )

    await callback.message.edit_text(
        "<b>📖 Как пользоваться ботом</b>\n\n"
        "<b>Шаг 2: ИИ-инструменты</b>\n\n"
        "Нажми «🤖 ИИ-инструменты»:\n"
        "• <b>Рерайт</b> — улучшить текст\n"
        "• <b>Хештеги</b> — подобрать теги\n"
        "• <b>A/B варианты</b> — 3 версии поста\n"
        "• <b>Копировать стиль</b> — писать как образец",
        reply_markup=tutorial_kb,
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "tutorial:3")
async def cb_tutorial_3(callback: types.CallbackQuery, state: FSMContext):
    """Туториал шаг 3"""
    tutorial_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Начать!", callback_data="start:begin")],
            [InlineKeyboardButton(text="← Назад", callback_data="tutorial:2")],
        ]
    )

    await callback.message.edit_text(
        "<b>📖 Как пользоваться ботом</b>\n\n"
        "<b>Шаг 3: Публикация</b>\n\n"
        "Готовый пост можно:\n"
        "• <b>Сохранить</b> в черновики\n"
        "• <b>Отправить</b> в Telegram-канал\n"
        "• <b>Отредактировать</b> с помощью ИИ\n\n"
        "Бот должен быть админом канала для публикации.\n\n"
        "<i>Готов начать? Жми кнопку!</i>",
        reply_markup=tutorial_kb,
    )
    await callback.answer()


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📝 Создать пост", callback_data="help:create"),
                InlineKeyboardButton(text="📂 Черновики", callback_data="help:drafts"),
            ],
            [
                InlineKeyboardButton(text="🤖 ИИ-инструменты", callback_data="help:ai"),
                InlineKeyboardButton(text="📅 Планирование", callback_data="help:plan"),
            ],
            [
                InlineKeyboardButton(text="📖 Все команды", callback_data="help:commands"),
            ],
        ]
    )

    await message.answer(
        "<b>👋 Привет! Я помогу с контентом для Telegram-канала.</b>\n\n"
        "Выбери раздел или используй кнопки внизу экрана 👇",
        reply_markup=help_kb,
    )
    # Отправляем также reply-клавиатуру
    await message.answer("Главное меню:", reply_markup=main_menu_kb)


@dp.callback_query(lambda c: c.data and c.data.startswith("help:"))
async def cb_help_section(callback: types.CallbackQuery, state: FSMContext):
    """Показать раздел справки"""
    section = callback.data.split(":")[1]

    if section == "create":
        text = (
            "<b>📝 Создание постов</b>\n\n"
            "<b>/idea</b> — ИИ сгенерирует идеи для постов по описанию канала, "
            "или напишет готовый пост по твоей идее\n\n"
            "<b>/draft</b> — пошаговое создание черновика: идея → заголовок → текст → заключение\n\n"
            "<b>/save_media</b> — сохранить фото/видео/кружок с подписью как черновик"
        )
    elif section == "drafts":
        text = (
            "<b>📂 Работа с черновиками</b>\n\n"
            "<b>/my_drafts</b> — список всех черновиков с пагинацией\n\n"
            "<b>/edit_draft</b> — редактировать черновик по номеру\n\n"
            "<b>/delete_draft</b> — удалить черновик\n\n"
            "<b>/send_draft</b> — отправить черновик в канал\n\n"
            "<b>/search</b> — найти черновик по ключевым словам"
        )
    elif section == "ai":
        text = (
            "<b>🤖 ИИ-инструменты</b>\n\n"
            "<b>/rewrite</b> — улучшить текст: ИИ сделает его живее и понятнее\n\n"
            "<b>/hashtags</b> — подобрать хештеги к посту\n\n"
            "<b>/variants</b> — сгенерировать 3 варианта поста для A/B теста\n\n"
            "<b>/style</b> — написать пост в стиле примера"
        )
    elif section == "plan":
        text = (
            "<b>📅 Планирование контента</b>\n\n"
            "<b>/plan</b> — сгенерировать контент-план на неделю или месяц\n\n"
            "<b>/templates</b> — готовые структуры постов: новость, обзор, история, совет, опрос"
        )
    elif section == "commands":
        text = (
            "<b>📖 Все команды</b>\n\n"
            "<b>Основные:</b>\n"
            "/start, /help, /cancel\n\n"
            "<b>Посты:</b>\n"
            "/idea, /draft, /save_media\n\n"
            "<b>Черновики:</b>\n"
            "/my_drafts, /edit_draft, /delete_draft, /send_draft, /search\n\n"
            "<b>ИИ:</b>\n"
            "/rewrite, /hashtags, /variants, /style\n\n"
            "<b>Планирование:</b>\n"
            "/plan, /templates"
        )
    else:
        text = "Раздел не найден."

    back_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="← Назад к справке", callback_data="help:back")]
        ]
    )

    await callback.message.edit_text(text, reply_markup=back_kb)
    await callback.answer()


@dp.callback_query(lambda c: c.data == "help:back")
async def cb_help_back(callback: types.CallbackQuery, state: FSMContext):
    """Вернуться к главной справке"""
    help_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📝 Создать пост", callback_data="help:create"),
                InlineKeyboardButton(text="📂 Черновики", callback_data="help:drafts"),
            ],
            [
                InlineKeyboardButton(text="🤖 ИИ-инструменты", callback_data="help:ai"),
                InlineKeyboardButton(text="📅 Планирование", callback_data="help:plan"),
            ],
            [
                InlineKeyboardButton(text="📖 Все команды", callback_data="help:commands"),
            ],
        ]
    )

    await callback.message.edit_text(
        "<b>👋 Привет! Я помогу с контентом для Telegram-канала.</b>\n\n"
        "Выбери раздел или используй кнопки внизу экрана 👇",
        reply_markup=help_kb,
    )
    await callback.answer()


# ----- КНОПКИ (Reply-клавиатура) -----


# ----- НОВОЕ ГЛАВНОЕ МЕНЮ С ПОДКАТЕГОРИЯМИ -----


def get_create_post_kb():
    """Подменю: Создать пост"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✨ Сгенерировать идеи", callback_data="menu:idea")],
            [InlineKeyboardButton(text="📝 Написать черновик", callback_data="menu:draft")],
            [InlineKeyboardButton(text="📎 Сохранить медиа", callback_data="menu:media")],
            [InlineKeyboardButton(text="← Назад", callback_data="menu:back")],
        ]
    )


def get_drafts_kb():
    """Подменю: Черновики"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📂 Все черновики", callback_data="menu:my_drafts")],
            [InlineKeyboardButton(text="🖼 Медиатека", callback_data="menu:media_gallery")],
            [InlineKeyboardButton(text="✏️ Редактировать", callback_data="menu:edit")],
            [InlineKeyboardButton(text="🗑 Удалить", callback_data="menu:delete")],
            [InlineKeyboardButton(text="📤 Отправить в канал", callback_data="menu:send")],
            [InlineKeyboardButton(text="← Назад", callback_data="menu:back")],
        ]
    )


def get_ai_tools_kb():
    """Подменю: ИИ-инструменты"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Рерайт текста", callback_data="menu:rewrite")],
            [InlineKeyboardButton(text="#️⃣ Генерация хештегов", callback_data="menu:hashtags")],
            [InlineKeyboardButton(text="🎯 A/B варианты", callback_data="menu:variants")],
            [InlineKeyboardButton(text="🎨 Копировать стиль", callback_data="menu:style")],
            [InlineKeyboardButton(text="🗜 Сократить / 📈 Расширить", callback_data="menu:shorten_expand")],
            [InlineKeyboardButton(text="← Назад", callback_data="menu:back")],
        ]
    )


def get_planning_kb():
    """Подменю: Планирование"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📅 Контент-план", callback_data="menu:plan")],
            [InlineKeyboardButton(text="📋 Шаблоны постов", callback_data="menu:templates")],
            [InlineKeyboardButton(text="← Назад", callback_data="menu:back")],
        ]
    )


@dp.message(lambda m: m.text == "📝 Создать пост")
async def btn_create_post(message: types.Message, state: FSMContext):
    """Показать подменю создания поста"""
    await message.answer(
        "<b>📝 Создать пост</b>\n\nВыбери, что хочешь сделать:",
        reply_markup=get_create_post_kb(),
    )


@dp.message(lambda m: m.text == "📂 Черновики")
async def btn_my_drafts_menu(message: types.Message, state: FSMContext):
    """Показать подменю черновиков"""
    await message.answer(
        "<b>📂 Черновики</b>\n\nВыбери действие:",
        reply_markup=get_drafts_kb(),
    )


@dp.message(lambda m: m.text == "🤖 ИИ-инструменты")
async def btn_ai_tools(message: types.Message, state: FSMContext):
    """Показать подменю ИИ-инструментов"""
    await message.answer(
        "<b>🤖 ИИ-инструменты</b>\n\nВыбери инструмент:",
        reply_markup=get_ai_tools_kb(),
    )


@dp.message(lambda m: m.text == "📅 Планирование")
async def btn_planning(message: types.Message, state: FSMContext):
    """Показать подменю планирования"""
    await message.answer(
        "<b>📅 Планирование</b>\n\nВыбери действие:",
        reply_markup=get_planning_kb(),
    )


@dp.message(lambda m: m.text == "🔍 Поиск")
async def btn_search(message: types.Message, state: FSMContext):
    """Начать поиск по черновикам"""
    await cmd_search(message, state)


@dp.message(lambda m: m.text == "❓ Помощь")
async def btn_help(message: types.Message):
    """Показать справку"""
    await cmd_help(message)


# ----- ОБРАБОТЧИКИ INLINE-МЕНЮ -----


@dp.callback_query(lambda c: c.data and c.data.startswith("menu:"))
async def cb_menu_action(callback: types.CallbackQuery, state: FSMContext):
    """Обработка нажатий на кнопки подменю"""
    action = callback.data.split(":")[1]

    # Сохраняем telegram_id пользователя в state для использования в командах
    await state.update_data(_user_telegram_id=callback.from_user.id)

    # Закрываем меню
    await callback.message.delete()

    if action == "back":
        await callback.message.answer("Главное меню 👇", reply_markup=main_menu_kb)
    elif action == "idea":
        await cmd_idea(callback.message, state)
    elif action == "draft":
        await cmd_draft(callback.message, state)
    elif action == "media":
        await cmd_save_media_draft(callback.message, state)
    elif action == "my_drafts":
        await show_drafts_page(callback.message, callback.from_user.id, page=0)
    elif action == "edit":
        await cmd_edit_draft(callback.message, state)
    elif action == "delete":
        await cmd_delete_draft(callback.message, state)
    elif action == "send":
        await cmd_send_draft(callback.message, state)
    elif action == "media_gallery":
        await cmd_media_gallery(callback.message, state)
    elif action == "rewrite":
        await cmd_rewrite(callback.message, state)
    elif action == "hashtags":
        await cmd_hashtags(callback.message, state)
    elif action == "variants":
        await cmd_variants(callback.message, state)
    elif action == "style":
        await cmd_style(callback.message, state)
    elif action == "shorten_expand":
        await callback.message.answer(
            "Эти действия доступны после генерации поста (кнопка ✏️ Редактировать → 📉/📈). "
            "Сначала сгенерируй пост через /idea или ИИ-инструменты.",
            reply_markup=main_menu_kb,
        )
    elif action == "plan":
        await cmd_plan(callback.message, state)
    elif action == "templates":
        await cmd_templates(callback.message, state)

    await callback.answer()


# ----- /send_draft -----


@dp.message(Command("send_draft"))
async def cmd_send_draft(message: types.Message, state: FSMContext):
    """
    Запускаем отправку черновика в канал.
    Сначала просим номер (как в /my_drafts), потом @канал или chat_id.
    """
    await state.set_state(SendDraftForm.waiting_for_number)
    await message.answer(
        "<b>Отправить черновик в канал</b>\n\n"
        "1) Напиши номер черновика (1, 2, 3 ...), как в списке /my_drafts.\n"
        "2) Затем пришли @username канала или его chat_id.\n\n"
        "Важно: бот должен быть админом канала, чтобы публиковать посты.\n"
        "Если передумал — /cancel."
    )


@dp.message(SendDraftForm.waiting_for_number)
async def process_send_draft_number(message: types.Message, state: FSMContext):
    """
    Получаем номер черновика, сохраняем текст, спрашиваем канал.
    """
    text = (message.text or "").strip()
    if text.lower() == "/cancel":
        return await cmd_cancel(message, state)
    if not text.isdigit():
        await message.answer("Номер должен быть числом. Пришли, пожалуйста, номер черновика (например: 2).")
        return

    draft_number = int(text)
    user_id = await get_user_id_from_context(message, state)
    drafts = await get_user_drafts_full(user_id)

    if draft_number < 1 or draft_number > len(drafts):
        await message.answer(
            "Черновик с таким номером не найден среди твоих.\n"
            "Проверь номер в /my_drafts и попробуй ещё раз, или напиши /cancel."
        )
        return

    draft = drafts[draft_number - 1]
    await state.update_data(draft_text=draft.draft_text, draft_number=draft_number)

    await state.set_state(SendDraftForm.waiting_for_channel)
    await message.answer(
        f"Черновик №{draft_number} выбран.\n\n"
        "Теперь пришли @username канала или его chat_id, куда отправить пост.\n"
        "Пример: @mychannel или -1001234567890."
    )


@dp.message(SendDraftForm.waiting_for_channel)
async def process_send_draft_channel(message: types.Message, state: FSMContext):
    """
    Получаем канал, пытаемся отправить туда текст.
    Поддерживает: черновик из БД, сгенерированный пост с прикреплённым медиа.
    """
    channel = (message.text or "").strip()
    data = await state.get_data()

    # Сгенерированный пост (из genpost_send)
    genpost_text = data.get("genpost_text")
    genpost_media = data.get("genpost_media")

    # Черновик из БД
    draft_text = data.get("draft_text", "")
    draft_number = data.get("draft_number")
    draft_id = data.get("draft_id")

    await state.clear()

    def should_strip_caption(text: str) -> bool:
        # Telegram: caption max ~1024 chars. Берём запас.
        return text and len(text) > 900

    try:
        if genpost_text:
            # Отправляем сгенерированный пост
            if genpost_media:
                mtype = genpost_media["type"]
                fid = genpost_media["file_id"]
                caption = genpost_text

                if should_strip_caption(caption):
                    await bot.send_message(chat_id=channel, text=caption)
                    caption = None

                if mtype == "photo":
                    await bot.send_photo(chat_id=channel, photo=fid, caption=caption)
                elif mtype == "video":
                    await bot.send_video(chat_id=channel, video=fid, caption=caption)
                elif mtype == "video_note":
                    await bot.send_video_note(chat_id=channel, video_note=fid)
                    # Кружки не поддерживают подпись, отправляем текст отдельно
                    if caption:
                        await bot.send_message(chat_id=channel, text=caption)
                elif mtype == "document":
                    await bot.send_document(chat_id=channel, document=fid, caption=caption)
                elif mtype == "voice":
                    await bot.send_voice(chat_id=channel, voice=fid, caption=caption)
                else:
                    await bot.send_message(chat_id=channel, text=genpost_text)
            else:
                await bot.send_message(chat_id=channel, text=genpost_text)

            await message.answer(f"Пост отправлен в канал {channel}!")
            return

        # Отправляем черновик из БД
        if not draft_text:
            await message.answer("Не получилось получить текст черновика. Попробуй ещё раз /send_draft.")
            return

        # Если это медиа-драфт (храним как MEDIA|type|file_id|caption)
        media_info = parse_media_draft(draft_text)

        if media_info:
            mtype = media_info["type"]
            fid = media_info["file_id"]
            caption = media_info["caption"] or None

            if caption and should_strip_caption(caption):
                await bot.send_message(chat_id=channel, text=caption)
                caption = None

            if mtype == "photo":
                await bot.send_photo(chat_id=channel, photo=fid, caption=caption)
            elif mtype == "video":
                await bot.send_video(chat_id=channel, video=fid, caption=caption)
            elif mtype == "video_note":
                await bot.send_video_note(chat_id=channel, video_note=fid)
            elif mtype == "document":
                await bot.send_document(chat_id=channel, document=fid, caption=caption)
            elif mtype == "voice":
                await bot.send_voice(chat_id=channel, voice=fid, caption=caption)
            else:
                await bot.send_message(chat_id=channel, text=draft_text)
        else:
            await bot.send_message(chat_id=channel, text=draft_text)

        label = f"Черновик №{draft_number}" if draft_number else "Черновик"
        await message.answer(
            f"{label} отправлен в канал {channel}.\n"
            "Не забудь, что у бота должны быть права на публикацию."
        )
    except TelegramForbiddenError as e:
        await message.answer(
            "Не удалось отправить сообщение: похоже, у бота нет прав в этом канале.\n"
            "Убедись, что бот — админ, и попробуй снова.\n\n"
            f"Детали: {e}"
        )
    except Exception as e:
        await message.answer(
            "Не получилось отправить сообщение в канал. "
            "Проверь @username / chat_id и попробуй ещё раз.\n\n"
            f"Детали: {e}"
        )


# Старт отправки из инлайн-кнопки
@dp.callback_query(lambda c: c.data == "start_send_draft")
async def cb_start_send_draft(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(_user_telegram_id=callback.from_user.id)
    await cmd_send_draft(callback.message, state)
    await callback.answer()

# ----- /edit_draft -----


@dp.message(Command("edit_draft"))
async def cmd_edit_draft(message: types.Message, state: FSMContext):
    """
    Запускаем диалог редактирования черновика.
    Пользователь указывает номер (как в /my_drafts), затем присылает новый текст.
    """
    await state.set_state(EditDraftForm.waiting_for_id)
    await message.answer(
        "<b>Редактирование черновика</b>\n\n"
        "Напиши номер черновика (1, 2, 3 ...), как в списке /my_drafts, который нужно изменить.\n\n"
        "Если передумал — напиши /cancel."
    )


@dp.message(EditDraftForm.waiting_for_id)
async def process_edit_draft_id(message: types.Message, state: FSMContext):
    """
    Получаем номер черновика, просим отправить новый текст.
    """
    text = (message.text or "").strip()
    if text.lower() == "/cancel":
        return await cmd_cancel(message, state)

    if not text.isdigit():
        await message.answer("Номер должен быть числом. Пришли, пожалуйста, номер черновика (например: 2).")
        return

    draft_number = int(text)
    user_id = await get_user_id_from_context(message, state)
    drafts = await get_user_drafts_full(user_id)

    if draft_number < 1 or draft_number > len(drafts):
        await message.answer(
            "Черновик с таким номером не найден среди твоих.\n"
            "Проверь номер в /my_drafts и попробуй ещё раз, или напиши /cancel."
        )
        return

    draft = drafts[draft_number - 1]
    await state.update_data(draft_id=draft.id, draft_number=draft_number, _user_telegram_id=user_id)

    await state.set_state(EditDraftForm.waiting_for_text)
    await message.answer(
        f"<b>Черновик №{draft_number}</b> выбран.\n\n"
        "Пришли новый текст черновика (полностью), я заменю старый целиком."
    )


@dp.message(EditDraftForm.waiting_for_text)
async def process_edit_draft_text(message: types.Message, state: FSMContext):
    """
    Принимаем новый текст черновика и обновляем запись.
    """
    new_text = (message.text or "").strip()
    if not new_text:
        await message.answer("Текст пустой. Пришли, пожалуйста, полный текст черновика.")
        return

    data = await state.get_data()
    draft_id = data.get("draft_id")
    draft_number = data.get("draft_number")

    if draft_id is None:
        await state.clear()
        await message.answer("Не получилось определить черновик. Попробуй ещё раз с команды /edit_draft.")
        return

    async with session_factory() as session:
        result = await session.execute(select(Draft).where(Draft.id == draft_id))
        draft = result.scalar_one_or_none()
        if not draft:
            await state.clear()
            await message.answer("Черновик не найден. Возможно, он был удалён. Посмотри актуальный список в /my_drafts.")
            return

        draft.draft_text = new_text
        await session.commit()

    await state.clear()
    await message.answer(
        f"Черновик №{draft_number} обновлён и сохранён.\n\n"
        f"<b>Новый текст:</b>\n{new_text}"
    )

# ----- УТИЛИТА ДЛЯ СБОРКИ ЧЕРНОВИКА -----


async def finalize_draft(message: types.Message, state: FSMContext, conclusion_text: str):
    """
    Собирает текст черновика и сохраняет его в БД. Используется для обычного шага и для кнопки 'Пропустить'.
    """
    data = await state.get_data()
    idea = data.get("idea", "")
    title = data.get("title", "")
    body = data.get("body", "")

    # Если пользователь отправил "-", считаем, что отдельного заключения нет
    if conclusion_text == "-":
        conclusion_text = ""

    # Собираем полный черновик аккуратно с переносами строк
    parts = []
    if idea:
        parts.append(f"Идея: {idea.strip()}")
    if title:
        parts.append(f"Заголовок: {title.strip()}")
    if body:
        parts.append("Текст:\n" + body.strip())
    if conclusion_text:
        parts.append("Заключение:\n" + conclusion_text.strip())

    draft_text = "\n\n".join(parts).strip()

    user_id = await get_user_id_from_context(message, state)
    await create_draft(
        telegram_id=user_id,
        idea_text=idea,
        draft_text=draft_text,
    )

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📤 Отправить в канал", callback_data="start_send_draft")]
        ]
    )

    await message.answer(
        "Черновик собран и сохранён в базе.\n\n"
        f"<b>Твой черновик целиком:</b>\n{draft_text}",
        reply_markup=kb,
    )

    await state.clear()


# ----- CALLBACKS ДЛЯ УДАЛЕНИЯ -----


@dp.callback_query(lambda c: c.data and c.data.startswith("delete_confirm:"))
async def cb_delete_confirm(callback: types.CallbackQuery, state: FSMContext):
    """
    Подтверждение удаления через кнопку.
    """
    try:
        draft_id = int(callback.data.split("delete_confirm:")[1])
    except Exception:
        await callback.answer("Не удалось понять, что удалять.", show_alert=True)
        return

    success = await delete_user_draft(callback.from_user.id, draft_id)
    await state.clear()

    if success:
        await callback.message.edit_text("Черновик удалён.")
    else:
        await callback.message.edit_text(
            "Не удалось удалить черновик. Возможно, он уже удалён. Проверь список /my_drafts."
        )

    await callback.answer()


@dp.callback_query(lambda c: c.data == "delete_cancel")
async def cb_delete_cancel(callback: types.CallbackQuery, state: FSMContext):
    """
    Отмена удаления через кнопку.
    """
    await state.clear()
    await callback.message.edit_text("Удаление отменено.")
    await callback.answer()


@dp.callback_query(lambda c: c.data == "draft_cancel")
async def cb_draft_cancel(callback: types.CallbackQuery, state: FSMContext):
    """
    Отмена сценария создания черновика.
    """
    await state.clear()
    await callback.message.edit_text("Создание черновика отменено.")
    await callback.answer()


@dp.callback_query(lambda c: c.data == "draft_skip_conclusion")
async def cb_draft_skip_conclusion(callback: types.CallbackQuery, state: FSMContext):
    """
    Пропустить заключение и собрать черновик.
    Работает только если мы на шаге заключения.
    """
    current = await state.get_state()
    if current != DraftForm.conclusion:
        await callback.answer("Сейчас нельзя пропустить заключение.", show_alert=True)
        return

    # Используем finalize_draft с пустым заключением
    class DummyMessage:
        from_user = callback.from_user

        def __init__(self, message):
            self._chat = message.chat

        async def answer(self, text, **kwargs):
            return await callback.message.answer(text, **kwargs)

        @property
        def chat(self):
            return self._chat

    dummy_message = DummyMessage(callback.message)
    await finalize_draft(dummy_message, state, conclusion_text="")
    await callback.answer()


@dp.callback_query(lambda c: c.data == "save_generated_post")
async def cb_save_generated_post(callback: types.CallbackQuery, state: FSMContext):
    """
    Сохраняем сгенерированный пост (идея + текст) в черновики.
    """
    data = await state.get_data()
    idea_text = (data.get("last_generated_idea") or "").strip()
    post_text = (data.get("last_generated_post") or "").strip()

    if not post_text:
        await state.clear()
        await callback.message.edit_text("Не удалось найти текст поста. Попробуй сгенерировать заново.")
        await callback.answer()
        return

    await create_draft(
        telegram_id=callback.from_user.id,
        idea_text=idea_text or "Идея не указана",
        draft_text=post_text,
    )

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📤 Отправить в канал", callback_data="start_send_draft")]
        ]
    )

    await state.clear()
    await callback.message.edit_text("Пост сохранён в черновики. Отправить в канал?", reply_markup=kb)
    await callback.answer()


@dp.callback_query(lambda c: c.data == "close_generated_post")
async def cb_close_generated_post(callback: types.CallbackQuery, state: FSMContext):
    """
    Закрыть карточку сгенерированного поста без сохранения.
    """
    await state.clear()
    await callback.message.edit_text("Ок, пост не сохранён.")
    await callback.answer()


# ----- РЕДАКТИРОВАНИЕ СГЕНЕРИРОВАННОГО ПОСТА -----


def _get_genpost_main_kb():
    """Клавиатура для сгенерированного поста."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="💾 Сохранить", callback_data="genpost_save"),
                InlineKeyboardButton(text="📤 В канал", callback_data="genpost_send"),
            ],
            [
                InlineKeyboardButton(text="✏️ Редактировать", callback_data="genpost_edit_menu"),
            ],
            [InlineKeyboardButton(text="❌ Закрыть", callback_data="genpost_close")],
        ]
    )


def _get_genpost_edit_kb():
    """Подменю редактирования."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🤖 Попросить ИИ изменить", callback_data="genpost_ai_edit")],
            [
                InlineKeyboardButton(text="📉 Сократить", callback_data="genpost_shorten"),
                InlineKeyboardButton(text="📈 Расширить", callback_data="genpost_expand"),
            ],
            [InlineKeyboardButton(text="📎 Прикрепить медиа", callback_data="genpost_attach_media")],
            [InlineKeyboardButton(text="✏️ Изменить заголовок (ИИ)", callback_data="genpost_ai_title")],
            [InlineKeyboardButton(text="#️⃣ Добавить хештеги", callback_data="genpost_add_hashtags")],
            [InlineKeyboardButton(text="← Назад", callback_data="genpost_back")],
        ]
    )


@dp.callback_query(lambda c: c.data == "genpost_close")
async def cb_genpost_close(callback: types.CallbackQuery, state: FSMContext):
    """Закрыть без сохранения."""
    await state.clear()
    await callback.message.edit_text("Ок, пост не сохранён.")
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_save")
async def cb_genpost_save(callback: types.CallbackQuery, state: FSMContext):
    """Сохранить сгенерированный пост в черновики."""
    data = await state.get_data()
    idea_text = data.get("last_generated_idea", "")
    post_text = data.get("last_generated_post", "")
    attached_media = data.get("attached_media")

    if not post_text:
        await state.clear()
        await callback.message.edit_text("Не удалось найти текст поста. Попробуй сгенерировать заново.")
        await callback.answer()
        return

    # Если есть прикреплённое медиа, сохраняем в формате MEDIA|...
    if attached_media:
        draft_text = f"MEDIA|{attached_media['type']}|{attached_media['file_id']}|{post_text}"
    else:
        draft_text = post_text

    await create_draft(
        telegram_id=callback.from_user.id,
        idea_text=idea_text or "Идея не указана",
        draft_text=draft_text,
    )

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📤 Отправить в канал", callback_data="start_send_draft")]
        ]
    )

    await state.clear()
    await callback.message.edit_text("Пост сохранён в черновики!", reply_markup=kb)
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_send")
async def cb_genpost_send(callback: types.CallbackQuery, state: FSMContext):
    """Отправить сгенерированный пост в канал."""
    data = await state.get_data()
    post_text = data.get("last_generated_post", "")
    attached_media = data.get("attached_media")

    if not post_text:
        await callback.answer("Нет текста для отправки.", show_alert=True)
        return

    # Сохраняем данные для отправки
    await state.update_data(
        genpost_text=post_text,
        genpost_media=attached_media,
    )
    await state.set_state(SendDraftForm.waiting_for_channel)

    await callback.message.answer(
        "Куда отправить пост?\n\n"
        "Пришли @username канала или его chat_id.\n"
        "Пример: @mychannel или -1001234567890.\n\n"
        "Если передумал — /cancel."
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_edit_menu")
async def cb_genpost_edit_menu(callback: types.CallbackQuery, state: FSMContext):
    """Показать меню редактирования."""
    await callback.message.edit_reply_markup(reply_markup=_get_genpost_edit_kb())
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_back")
async def cb_genpost_back(callback: types.CallbackQuery, state: FSMContext):
    """Вернуться к основному меню поста."""
    data = await state.get_data()
    post_text = data.get("last_generated_post", "")
    attached_media = data.get("attached_media")

    media_info = ""
    if attached_media:
        media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"

    await callback.message.edit_text(
        f"<b>Готовый пост:</b>\n\n{post_text}{media_info}",
        reply_markup=_get_genpost_main_kb(),
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_ai_edit")
async def cb_genpost_ai_edit(callback: types.CallbackQuery, state: FSMContext):
    """Попросить ИИ изменить пост."""
    await state.set_state(EditGeneratedPostForm.waiting_for_ai_edit)
    await callback.message.answer(
        "Напиши, что нужно изменить в посте.\n"
        "Например: «добавь больше примеров», «сделай короче», «измени тон на более дружелюбный».\n\n"
        "Если передумал — /cancel."
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_ai_title")
async def cb_genpost_ai_title(callback: types.CallbackQuery, state: FSMContext):
    """Попросить ИИ добавить/изменить заголовок."""
    data = await state.get_data()
    post_text = data.get("last_generated_post", "")

    if not post_text:
        await callback.answer("Нет текста поста.", show_alert=True)
        return

    await callback.message.answer("Генерирую заголовок...")

    edited = await edit_post_with_ai(post_text, "Добавь цепляющий заголовок в начало поста (1 строка, выделенный). Если заголовок уже есть — улучши его.")

    if not edited:
        await callback.message.answer("Не удалось сгенерировать заголовок. Попробуй ещё раз.")
        await callback.answer()
        return

    await state.update_data(last_generated_post=edited)

    media_info = ""
    attached_media = data.get("attached_media")
    if attached_media:
        media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"

    await callback.message.answer(
        f"<b>Обновлённый пост:</b>\n\n{edited}{media_info}",
        reply_markup=_get_genpost_main_kb(),
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_shorten")
async def cb_genpost_shorten(callback: types.CallbackQuery, state: FSMContext):
    """Сократить пост."""
    data = await state.get_data()
    post_text = data.get("last_generated_post", "")

    if not post_text:
        await callback.answer("Нет текста поста.", show_alert=True)
        return

    await callback.message.answer("Сокращаю пост...")

    edited = await edit_post_with_ai(post_text, "Сократи этот пост примерно в 2 раза, сохрани главную мысль и структуру.")

    if not edited:
        await callback.message.answer("Не удалось сократить. Попробуй ещё раз.")
        await callback.answer()
        return

    await state.update_data(last_generated_post=edited)

    media_info = ""
    attached_media = data.get("attached_media")
    if attached_media:
        media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"

    await callback.message.answer(
        f"<b>Сокращённый пост:</b>\n\n{edited}{media_info}",
        reply_markup=_get_genpost_main_kb(),
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_expand")
async def cb_genpost_expand(callback: types.CallbackQuery, state: FSMContext):
    """Расширить пост."""
    data = await state.get_data()
    post_text = data.get("last_generated_post", "")

    if not post_text:
        await callback.answer("Нет текста поста.", show_alert=True)
        return

    await callback.message.answer("Расширяю пост...")

    edited = await edit_post_with_ai(post_text, "Расширь этот пост: добавь больше деталей, примеров и аргументов. Увеличь объём примерно в 1.5-2 раза.")

    if not edited:
        await callback.message.answer("Не удалось расширить. Попробуй ещё раз.")
        await callback.answer()
        return

    await state.update_data(last_generated_post=edited)

    media_info = ""
    attached_media = data.get("attached_media")
    if attached_media:
        media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"

    await callback.message.answer(
        f"<b>Расширенный пост:</b>\n\n{edited}{media_info}",
        reply_markup=_get_genpost_main_kb(),
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_add_hashtags")
async def cb_genpost_add_hashtags(callback: types.CallbackQuery, state: FSMContext):
    """Добавить хештеги к посту."""
    data = await state.get_data()
    post_text = data.get("last_generated_post", "")

    if not post_text:
        await callback.answer("Нет текста поста.", show_alert=True)
        return

    await callback.message.answer("Подбираю хештеги...")

    hashtags = await generate_hashtags_with_ai(post_text)

    if not hashtags:
        await callback.message.answer("Не удалось подобрать хештеги. Попробуй ещё раз.")
        await callback.answer()
        return

    # Добавляем хештеги в конец поста
    new_post = f"{post_text}\n\n{hashtags}"
    await state.update_data(last_generated_post=new_post)

    media_info = ""
    attached_media = data.get("attached_media")
    if attached_media:
        media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"

    await callback.message.answer(
        f"<b>Пост с хештегами:</b>\n\n{new_post}{media_info}",
        reply_markup=_get_genpost_main_kb(),
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "genpost_attach_media")
async def cb_genpost_attach_media(callback: types.CallbackQuery, state: FSMContext):
    """Прикрепить медиа к посту."""
    await state.set_state(EditGeneratedPostForm.waiting_for_media)
    await callback.message.answer(
        "Пришли фото, видео, кружок, документ или голосовое сообщение.\n"
        "Оно будет прикреплено к посту.\n\n"
        "Если передумал — /cancel."
    )
    await callback.answer()


@dp.message(EditGeneratedPostForm.waiting_for_ai_edit)
async def process_genpost_ai_edit(message: types.Message, state: FSMContext):
    """Получаем запрос на редактирование и отправляем ИИ."""
    if (message.text or "").strip().lower() == "/cancel":
        # Не сбрасываем пост, а возвращаемся к меню редактирования
        await state.set_state(EditGeneratedPostForm.editing)
        data = await state.get_data()
        post_text = data.get("last_generated_post", "")
        attached_media = data.get("attached_media")
        media_info = ""
        if attached_media:
            media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"
        await message.answer(
            f"Редактирование отменено.\n\n<b>Готовый пост:</b>\n\n{post_text}{media_info}",
            reply_markup=_get_genpost_main_kb(),
        )
        return

    edit_request = (message.text or "").strip()
    if not edit_request:
        await message.answer("Пустой запрос. Напиши, что нужно изменить в посте.")
        return

    data = await state.get_data()
    post_text = data.get("last_generated_post", "")

    if not post_text:
        await state.clear()
        await message.answer("Не нашёл текст поста. Попробуй сгенерировать заново через /idea.")
        return

    await message.answer("Редактирую пост...")

    edited = await edit_post_with_ai(post_text, edit_request)

    if not edited:
        await message.answer("Не удалось отредактировать пост. Попробуй ещё раз или сформулируй запрос по-другому.")
        await state.set_state(EditGeneratedPostForm.editing)
        return

    await state.update_data(last_generated_post=edited)
    await state.set_state(EditGeneratedPostForm.editing)

    media_info = ""
    attached_media = data.get("attached_media")
    if attached_media:
        media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"

    await message.answer(
        f"<b>Обновлённый пост:</b>\n\n{edited}{media_info}",
        reply_markup=_get_genpost_main_kb(),
    )


@dp.message(EditGeneratedPostForm.waiting_for_media)
async def process_genpost_attach_media(message: types.Message, state: FSMContext):
    """Прикрепляем медиа к сгенерированному посту."""
    if (message.text or "").strip().lower() == "/cancel":
        # Не сбрасываем пост, а возвращаемся к меню редактирования
        await state.set_state(EditGeneratedPostForm.editing)
        data = await state.get_data()
        post_text = data.get("last_generated_post", "")
        attached_media = data.get("attached_media")
        media_info = ""
        if attached_media:
            media_info = f"\n\n📎 Прикреплено: {attached_media['type']}"
        await message.answer(
            f"Прикрепление отменено.\n\n<b>Готовый пост:</b>\n\n{post_text}{media_info}",
            reply_markup=_get_genpost_main_kb(),
        )
        return

    media_type = None
    file_id = None

    if message.photo:
        media_type = "photo"
        file_id = message.photo[-1].file_id
    elif message.video:
        media_type = "video"
        file_id = message.video.file_id
    elif message.video_note:
        media_type = "video_note"
        file_id = message.video_note.file_id
    elif message.document:
        media_type = "document"
        file_id = message.document.file_id
    elif message.voice:
        media_type = "voice"
        file_id = message.voice.file_id

    if not media_type or not file_id:
        await message.answer("Не вижу медиа. Пришли фото, видео, кружок, документ или голосовое.")
        return

    await state.update_data(attached_media={"type": media_type, "file_id": file_id})
    await state.set_state(EditGeneratedPostForm.editing)

    data = await state.get_data()
    post_text = data.get("last_generated_post", "")

    await message.answer(
        f"<b>Готовый пост:</b>\n\n{post_text}\n\n📎 Прикреплено: {media_type}",
        reply_markup=_get_genpost_main_kb(),
    )


# ----- СОХРАНЕНИЕ МЕДИА ЧЕРНОВИКА -----


def parse_media_draft(draft_text: str):
    """
    Формат хранения медиа-драфта:
    MEDIA|type|file_id|caption
    type: photo, video, video_note, document, voice
    """
    if not draft_text.startswith("MEDIA|"):
        return None
    parts = draft_text.split("|", 3)
    if len(parts) < 4:
        return None
    return {
        "type": parts[1],
        "file_id": parts[2],
        "caption": parts[3],
    }


@dp.message(Command("save_media_draft"))
async def cmd_save_media_draft(message: types.Message, state: FSMContext):
    """
    Просим пользователя прислать медиа (фото/видео/видеозаметку/док/войс) с подписью.
    """
    await state.set_state(SaveMediaDraftForm.waiting_for_media)
    await message.answer(
        "<b>Сохранение медиа в черновик</b>\n\n"
        "Пришли фото, видео, видеозаметку (кружок), документ или голосовое сообщение.\n"
        "Добавь подпись — она попадёт в черновик.\n"
        "Если передумал — /cancel."
    )


@dp.message(SaveMediaDraftForm.waiting_for_media)
async def process_save_media_draft(message: Message, state: FSMContext):
    """
    Принимаем медиа, сохраняем file_id + подпись в черновик.
    """
    # Если пользователь передумал и отправил /cancel — выходим в общий обработчик
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    caption = message.caption or ""

    media_type = None
    file_id = None

    if message.photo:
        media_type = "photo"
        file_id = message.photo[-1].file_id  # лучшее качество
    elif message.video:
        media_type = "video"
        file_id = message.video.file_id
    elif message.video_note:
        media_type = "video_note"
        file_id = message.video_note.file_id
    elif message.document:
        media_type = "document"
        file_id = message.document.file_id
    elif message.voice:
        media_type = "voice"
        file_id = message.voice.file_id

    if not media_type or not file_id:
        await message.answer(
            "Не вижу медиа. Пришли фото, видео, кружок, документ или голосовое сообщение."
        )
        return

    payload = f"MEDIA|{media_type}|{file_id}|{caption}"

    user_id = await get_user_id_from_context(message, state)
    await create_draft(
        telegram_id=user_id,
        idea_text=caption or "Медиа без подписи",
        draft_text=payload,
    )

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📤 Отправить в канал", callback_data="start_send_draft")]
        ]
    )

    await state.clear()
    await message.answer("Медиа сохранено в черновики. Отправить в канал?", reply_markup=kb)


@dp.callback_query(lambda c: c.data == "idea_mode:channel")
async def cb_idea_mode_channel(callback: types.CallbackQuery, state: FSMContext):
    """
    Ветвь /idea: генерируем идеи для канала.
    """
    await state.set_state(PlanForm.profile)
    await callback.message.answer(
        "Опиши свой канал: тематику, аудиторию, стиль.\n"
        "Например: \"Канал про IT-новости для новичков, стиль дружелюбный и простой.\""
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "idea_mode:own")
async def cb_idea_mode_own(callback: types.CallbackQuery, state: FSMContext):
    """
    Ветвь /idea: у пользователя уже есть своя идея поста.
    """
    await state.set_state(PlanForm.own_idea)
    await callback.message.answer(
        "<b>Твоя идея поста</b>\n\n"
        "Пришли текст идеи/темы поста, с которой хочешь работать.\n"
        "Например: \"Как я за месяц улучшил вовлечённость в канале\"."
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "ownidea_to_draft")
async def cb_ownidea_to_draft(callback: types.CallbackQuery, state: FSMContext):
    """
    Пользователь выбрал собрать черновик по своей идее (ветка /idea).
    Переключаемся в FSM черновиков, пропуская шаг с вводом идеи.
    """
    data = await state.get_data()
    idea_text = (data.get("idea_for_draft") or "").strip()

    if not idea_text:
        await state.clear()
        await callback.answer("Не получилось получить идею. Попробуй ещё раз через /idea.", show_alert=True)
        return

    # Переходим в FSM DraftForm, сразу на шаг заголовка,
    # сохраняя идею в состоянии.
    await state.set_state(DraftForm.title)
    await state.update_data(idea=idea_text)

    await callback.message.answer(
        f"Делаем черновик по идее:\n\n<code>{idea_text}</code>\n\n"
        "<b>Шаг 2. Заголовок</b>\n\n"
        "Теперь придумай и пришли заголовок поста.\n"
        "Подсказка: сделай его коротким и конкретным, можно с результатом или выгодой для читателя.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="draft_cancel")]
            ]
        ),
    )

    await state.update_data(idea_for_draft=None)
    await callback.answer()


@dp.callback_query(lambda c: c.data == "ownidea_self")
async def cb_ownidea_self(callback: types.CallbackQuery, state: FSMContext):
    """
    Пользователь выбрал писать пост сам, без черновика от бота.
    """
    await state.clear()
    await callback.message.edit_text(
        "Ок, пиши пост сам.\n"
        "Если захочешь, я могу помочь собрать структуру через /draft или кнопки внизу.",
    )
    await callback.answer()


# ----- ИИ-ГЕНЕРАЦИЯ ПОЛНОГО ПОСТА ПО ИДЕЕ -----


def _generate_post_sync(idea_text: str) -> str:
    """
    Синхронный вызов OpenAI для генерации полного поста по идее.
    Если ключа нет или произошла ошибка, возвращает пустую строку.
    """
    if not openai_client:
        print("OPENAI_API_KEY is not set, cannot generate full post.")
        return ""

    system_message = (
        "Ты автор постов для Telegram-каналов. "
        "Пиши на русском, структурировано и живо."
    )

    user_prompt = (
        "Напиши полный текст поста по идее ниже.\n\n"
        f"Идея: {idea_text}\n\n"
        "Требования:\n"
        "- В самом начале поста сделай цепляющий заголовок (1 строка, можно с эмодзи).\n"
        "- Стиль: живой, понятный широкой аудитории, без канцелярита.\n"
        "- Структура: заголовок, вступление, 2–4 абзаца основной части, короткое заключение с выводом/призывом.\n"
        "- Без служебных фраз вроде «вот ваш текст», сразу сам пост.\n"
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        print("GPT error in full-post generation:", repr(e))
        return ""


async def generate_full_post_with_ai(idea_text: str) -> str:
    """
    Асинхронная обёртка для синхронного вызова OpenAI.
    """
    return await to_thread(_generate_post_sync, idea_text)


def _edit_post_with_ai_sync(current_post: str, edit_request: str) -> str:
    """
    Синхронный вызов OpenAI для редактирования/дополнения поста.
    """
    if not openai_client:
        print("OPENAI_API_KEY is not set, cannot edit post.")
        return ""

    system_message = (
        "Ты редактор постов для Telegram-каналов. "
        "Пользователь даёт тебе текущий текст поста и просьбу, что изменить. "
        "Верни отредактированный пост целиком."
    )

    user_prompt = (
        "Текущий текст поста:\n"
        f"---\n{current_post}\n---\n\n"
        f"Запрос пользователя: {edit_request}\n\n"
        "Требования:\n"
        "- Верни отредактированный пост целиком.\n"
        "- Сохрани стиль и структуру, если пользователь не просит другое.\n"
        "- Без служебных фраз вроде «вот отредактированный текст», сразу сам пост.\n"
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        print("GPT error in post editing:", repr(e))
        return ""


async def edit_post_with_ai(current_post: str, edit_request: str) -> str:
    """
    Асинхронная обёртка для редактирования поста через ИИ.
    """
    return await to_thread(_edit_post_with_ai_sync, current_post, edit_request)


@dp.callback_query(lambda c: c.data == "ownidea_generate_post")
async def cb_ownidea_generate_post(callback: types.CallbackQuery, state: FSMContext):
    """
    Пользователь просит ИИ написать полный пост по его идее.
    """
    data = await state.get_data()
    idea_text = (data.get("idea_for_draft") or "").strip()

    if not idea_text:
        await state.clear()
        await callback.answer("Не получилось получить идею. Попробуй ещё раз через /idea.", show_alert=True)
        return

    await callback.message.answer("Пишу пост по твоей идее, подожди несколько секунд...")

    post_text = await generate_full_post_with_ai(idea_text)

    if not post_text:
        await state.clear()
        await callback.message.answer(
            "Не удалось сгенерировать пост (нет ключа или ошибка ИИ). "
            "Попробуй ещё раз или собери черновик вручную через /draft."
        )
        await callback.answer()
        return

    # Сохраняем сгенерированный пост в состоянии
    await state.update_data(
        last_generated_idea=idea_text,
        last_generated_post=post_text,
        attached_media=None,  # для прикрепления медиа
    )
    await state.set_state(EditGeneratedPostForm.editing)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="💾 Сохранить", callback_data="genpost_save"),
                InlineKeyboardButton(text="📤 В канал", callback_data="genpost_send"),
            ],
            [
                InlineKeyboardButton(text="✏️ Редактировать", callback_data="genpost_edit_menu"),
            ],
            [InlineKeyboardButton(text="❌ Закрыть", callback_data="genpost_close")],
        ]
    )

    await callback.message.answer(
        f"<b>Готовый пост:</b>\n\n{post_text}",
        reply_markup=kb,
    )

    await callback.answer()


# ----- /idea -----

@dp.message(Command("idea"))
async def cmd_idea(message: types.Message, state: FSMContext):
    """
    Обработчик команды /idea.
    Предлагаем два варианта работы:
    1) Сгенерировать идеи для канала.
    2) У пользователя уже есть идея поста, и он хочет работать с ней.
    """
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✨ Идеи для канала", callback_data="idea_mode:channel"
                )
            ],
            [
                InlineKeyboardButton(
                    text="💡 У меня уже есть идея", callback_data="idea_mode:own"
                )
            ],
        ]
    )
    await message.answer(
        "Как будем работать с идеями?\n\n"
        "✨ Идеи для канала — ты описываешь канал, я предложу варианты постов.\n"
        "💡 У меня уже есть идея — ты присылаешь свою тему, и дальше решаем, как с ней работать.",
        reply_markup=kb,
    )


@dp.message(PlanForm.profile)
async def process_profile(message: types.Message, state: FSMContext):
    """
    Здесь мы получаем описание канала от пользователя,
    вызываем LangGraph (с заглушкой) и отдаём идеи.
    """
    profile_text = message.text

    await message.answer("Генерирую идеи постов, подожди несколько секунд...")

    # Вызываем граф в отдельном потоке, чтобы не блокировать бота
    result = await to_thread(
        plan_graph.invoke,
        {"profile": profile_text, "ideas": []}
    )

    ideas = result["ideas"]

    if not ideas:
        await message.answer("Не удалось сгенерировать идеи. Попробуй описать канал по-другому.")
        await state.clear()
        return

    text = "Вот идеи постов для твоего канала:\n\n" + "\n".join(f"- {idea}" for idea in ideas)

    await message.answer(text)

    # Сбрасываем состояние — диалог завершён
    await state.clear()


@dp.message(PlanForm.own_idea)
async def process_own_idea_for_idea(message: types.Message, state: FSMContext):
    """
    Ветвь /idea, когда у пользователя уже есть своя идея поста.
    Мы фиксируем идею и предлагаем либо собрать по ней черновик, либо писать самому.
    """
    idea_text = (message.text or "").strip()
    if not idea_text:
        await message.answer("Идея пуста. Пришли, пожалуйста, текст идеи поста.")
        return

    await state.update_data(idea_for_draft=idea_text)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🤖 Написать пост по идее (ИИ)",
                    callback_data="ownidea_generate_post",
                )
            ],
            [
                InlineKeyboardButton(
                    text="📝 Собрать черновик по этой идее",
                    callback_data="ownidea_to_draft",
                )
            ],
            [
                InlineKeyboardButton(
                    text="✍ Я напишу пост сам",
                    callback_data="ownidea_self",
                )
            ],
        ]
    )

    await message.answer(
        f"Твоя идея поста:\n\n<code>{idea_text}</code>\n\n"
        "Выбирай, как поступить:\n"
        "🤖 ИИ напишет полный пост по идее;\n"
        "📝 Соберём черновик по шагам (как /draft);\n"
        "✍ Напишешь сам.\n\n"
        "Выбери, как двигаемся дальше:",
        reply_markup=kb,
    )


# ----- /draft -----

@dp.message(Command("draft"))
async def cmd_draft(message: types.Message, state: FSMContext):
    """
    Обработчик команды /draft.
    Объясняем механику и просим подтвердить старт пошагового диалога.
    """
    await state.set_state(DraftForm.confirm)
    await message.answer(
        "Я помогу тебе собрать черновик поста по шагам.\n\n"
        "Важно: текст поста будешь писать ТЫ, а я только подскажу, какие блоки заполнить.\n\n"
        "Если хочешь начать, отправь в ответ <b>+</b>.\n"
        "Если передумал — напиши /cancel."
    )


@dp.message(DraftForm.confirm)
async def process_draft_confirm(message: types.Message, state: FSMContext):
    """
    Ждём, пока пользователь явно подтвердит старт диалога с помощью "+".
    """
    text = (message.text or "").strip()

    if text != "+":
        await message.answer(
            "Чтобы начать работу над черновиком, отправь, пожалуйста, знак плюс: <b>+</b>.\n"
            "Если не хочешь продолжать, в любой момент можно написать /cancel."
        )
        return

    await state.set_state(DraftForm.idea)
    await message.answer(
        "<b>Шаг 1. Идея поста</b>\n\n"
        "Коротко опиши, о чём будет пост.\n"
        "Например: \"Как я за месяц улучшил продуктивность на учёбе\".",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="draft_cancel")]
            ]
        ),
    )


@dp.message(DraftForm.idea)
async def process_draft_idea(message: types.Message, state: FSMContext):
    """
    Шаг 1: получаем идею поста.
    """
    idea_text = (message.text or "").strip()

    if not idea_text:
        await message.answer("Идея пуста. Отправь, пожалуйста, короткое описание идеи поста.")
        return

    await state.update_data(idea=idea_text)

    await state.set_state(DraftForm.title)
    await message.answer(
        "<b>Шаг 2. Заголовок</b>\n\n"
        "Теперь придумай и пришли заголовок поста.\n"
        "Подсказка: сделай его коротким и конкретным, можно с результатом или выгодой для читателя.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="draft_cancel")]
            ]
        ),
    )


@dp.message(DraftForm.title)
async def process_draft_title(message: types.Message, state: FSMContext):
    """
    Шаг 2: получаем заголовок поста.
    """
    title_text = (message.text or "").strip()

    if not title_text:
        await message.answer("Заголовок пустой. Пришли, пожалуйста, текст заголовка.")
        return

    await state.update_data(title=title_text)

    await state.set_state(DraftForm.body)
    await message.answer(
        "<b>Шаг 3. Основной текст</b>\n\n"
        "Пришли основной текст поста: 1–3 абзаца.\n"
        "Можно описать шаги, историю, советы — всё, что раскрывает идею.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="❌ Отмена", callback_data="draft_cancel")]
            ]
        ),
    )


@dp.message(DraftForm.body)
async def process_draft_body(message: types.Message, state: FSMContext):
    """
    Шаг 3: получаем основной текст поста.
    """
    body_text = (message.text or "").strip()

    if not body_text:
        await message.answer("Текст пустой. Пришли, пожалуйста, основной текст поста.")
        return

    await state.update_data(body=body_text)

    await state.set_state(DraftForm.conclusion)
    await message.answer(
        "<b>Шаг 4. Заключение</b>\n\n"
        "Теперь пришли заключение или призыв к действию (1–3 предложения).\n"
        "Если не хочешь делать отдельное заключение, просто отправь <b>-</b>.\n"
        "Или нажми кнопку \"Пропустить\".",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="⏭ Пропустить заключение", callback_data="draft_skip_conclusion"
                    )
                ],
                [InlineKeyboardButton(text="❌ Отмена", callback_data="draft_cancel")],
            ]
        ),
    )


@dp.message(DraftForm.conclusion)
async def process_draft_conclusion(message: types.Message, state: FSMContext):
    """
    Шаг 4: получаем заключение и собираем финальный текст черновика.
    """
    conclusion_text = (message.text or "").strip()

    await finalize_draft(message, state, conclusion_text)


# ----- /my_drafts с пагинацией -----


def get_draft_actions_kb(draft_idx: int):
    """Кнопки действий для одного черновика"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✏️", callback_data=f"draft_act:edit:{draft_idx}"),
                InlineKeyboardButton(text="🗑", callback_data=f"draft_act:delete:{draft_idx}"),
                InlineKeyboardButton(text="📤", callback_data=f"draft_act:send:{draft_idx}"),
            ]
        ]
    )


def get_pagination_kb(page: int, total_pages: int):
    """Кнопки пагинации"""
    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton(text="← Назад", callback_data=f"drafts_page:{page - 1}"))
    buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="drafts_page:noop"))
    if page < total_pages - 1:
        buttons.append(InlineKeyboardButton(text="Вперёд →", callback_data=f"drafts_page:{page + 1}"))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


async def show_drafts_page(message_or_callback, telegram_id: int, page: int = 0, edit: bool = False):
    """Показать страницу черновиков с пагинацией (аккуратное форматирование)"""
    rows = await get_user_drafts_full(telegram_id)

    if not rows:
        text = "У тебя пока нет сохранённых черновиков."
        if edit and hasattr(message_or_callback, 'edit_text'):
            await message_or_callback.edit_text(text)
        else:
            target = message_or_callback.message if hasattr(message_or_callback, 'message') else message_or_callback
            await target.answer(text)
        return

    total = len(rows)
    total_pages = (total + DRAFTS_PER_PAGE - 1) // DRAFTS_PER_PAGE
    page = max(0, min(page, total_pages - 1))

    start_idx = page * DRAFTS_PER_PAGE
    end_idx = min(start_idx + DRAFTS_PER_PAGE, total)
    page_drafts = rows[start_idx:end_idx]

    lines = [f"<b>📂 Твои черновики</b> ({total} шт.)", ""]

    for i, row in enumerate(page_drafts):
        idx = start_idx + i + 1
        draft_text = (row.draft_text or "").strip()
        media_info = parse_media_draft(draft_text)

        if media_info:
            mtype = media_info["type"]
            caption = (media_info["caption"] or "—").strip()
            preview = f"📎 {mtype}\n{caption}"
        else:
            preview = draft_text
            if len(preview) > 500:
                preview = preview[:500].rstrip() + "..."

        lines.append(f"<b>#{idx}</b>")
        lines.append(preview)
        lines.append("────────────")
        lines.append("")  # пустая строка для отступа

    text = "\n".join(lines).strip()

    # Пагинация + быстрые действия
    buttons = []

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="← Назад", callback_data=f"drafts_page:{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="drafts_page:noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton(text="Вперёд →", callback_data=f"drafts_page:{page + 1}"))
    buttons.append(nav_buttons)

    buttons.append([
        InlineKeyboardButton(text="✏️ Редактировать", callback_data="quick:edit"),
        InlineKeyboardButton(text="🗑 Удалить", callback_data="quick:delete"),
    ])
    buttons.append([
        InlineKeyboardButton(text="📤 Отправить", callback_data="quick:send"),
        InlineKeyboardButton(text="🔄 Обновить", callback_data=f"drafts_page:{page}"),
    ])

    kb = InlineKeyboardMarkup(inline_keyboard=buttons)

    if edit and hasattr(message_or_callback, 'edit_text'):
        await message_or_callback.edit_text(text, reply_markup=kb)
    else:
        target = message_or_callback.message if hasattr(message_or_callback, 'message') else message_or_callback
        await target.answer(text, reply_markup=kb)


@dp.message(Command("my_drafts"))
async def cmd_my_drafts(message: types.Message, state: FSMContext = None):
    """Показываем черновики с пагинацией"""
    user_id = await get_user_id_from_context(message, state)
    await show_drafts_page(message, user_id, page=0)


@dp.callback_query(lambda c: c.data and c.data.startswith("drafts_page:"))
async def cb_drafts_page(callback: types.CallbackQuery, state: FSMContext):
    """Переключение страниц черновиков"""
    page_str = callback.data.split(":")[1]
    if page_str == "noop":
        await callback.answer()
        return
    page = int(page_str)
    await show_drafts_page(callback.message, callback.from_user.id, page=page, edit=True)
    await callback.answer()


@dp.callback_query(lambda c: c.data and c.data.startswith("quick:"))
async def cb_quick_action(callback: types.CallbackQuery, state: FSMContext):
    """Быстрые действия из списка черновиков"""
    action = callback.data.split(":")[1]
    
    # Сохраняем user_id для последующих команд
    await state.update_data(_user_telegram_id=callback.from_user.id)
    
    await callback.message.delete()

    if action == "edit":
        await cmd_edit_draft(callback.message, state)
    elif action == "delete":
        await cmd_delete_draft(callback.message, state)
    elif action == "send":
        await cmd_send_draft(callback.message, state)

    await callback.answer()


# ----- /delete_draft -----


@dp.message(Command("delete_draft"))
async def cmd_delete_draft(message: types.Message, state: FSMContext):
    """
    Запускаем диалог удаления черновика.
    Сначала просим пользователя указать номер черновика (как в /my_drafts).
    """
    await state.set_state(DeleteDraftForm.waiting_for_id)
    await message.answer(
        "<b>Удаление черновика</b>\n\n"
        "Напиши номер черновика (1, 2, 3 ...), как в списке /my_drafts.\n\n"
        "Если передумал — напиши /cancel."
    )


@dp.message(DeleteDraftForm.waiting_for_id)
async def process_delete_draft_id(message: types.Message, state: FSMContext):
    """
    Получаем от пользователя номер черновика, показываем краткую информацию и просим подтверждение.
    """
    text = (message.text or "").strip()
    if text.lower() == "/cancel":
        return await cmd_cancel(message, state)

    if not text.isdigit():
        await message.answer("Номер должен быть числом. Пришли, пожалуйста, номер черновика (например: 2).")
        return

    draft_number = int(text)
    user_id = await get_user_id_from_context(message, state)
    drafts = await get_user_drafts_full(user_id)

    if draft_number < 1 or draft_number > len(drafts):
        await message.answer(
            "Черновик с таким номером не найден среди твоих.\n"
            "Проверь номер в /my_drafts и попробуй ещё раз, или напиши /cancel."
        )
        return

    draft = drafts[draft_number - 1]

    idea_text = draft.idea_text
    draft_text = draft.draft_text

    await state.update_data(draft_id=draft.id, draft_number=draft_number)
    await state.set_state(DeleteDraftForm.waiting_for_confirm)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Удалить",
                    callback_data=f"delete_confirm:{draft.id}",
                )
            ],
            [InlineKeyboardButton(text="❌ Отмена", callback_data="delete_cancel")],
        ]
    )

    await message.answer(
        f"<b>Удаление черновика №{draft_number}</b>\n\n"
        f"Идея:\n{idea_text}\n\n"
        f"Текст:\n{draft_text}\n\n"
        "Подтверди действие кнопкой ниже.",
        reply_markup=kb,
    )


@dp.message(DeleteDraftForm.waiting_for_confirm)
async def process_delete_draft_confirm(message: types.Message, state: FSMContext):
    """
    Если пользователь решил написать текстом вместо кнопок — подсказываем про кнопки.
    """
    await message.answer("Нажми, пожалуйста, кнопку ниже: ✅ Удалить или ❌ Отмена.")
    

# ----- /cancel -----

@dp.message(Command("cancel"))
async def cmd_cancel(message: types.Message, state: FSMContext):
    """
    Отменяет любой текущий диалог (для /idea, /draft и т.п.).
    """
    # Сбрасываем состояние
    await state.clear()
    await message.answer(
        "Текущий диалог отменён. Можешь начать заново, например с /help или другой команды.",
        reply_markup=main_menu_kb,
    )


# =============================================
# НОВЫЕ ФУНКЦИИ: РЕРАЙТ, ХЕШТЕГИ, ВАРИАНТЫ И Т.Д.
# =============================================


# ----- ИИ-функции -----


def _rewrite_text_sync(original_text: str) -> str:
    """Синхронный рерайт текста через OpenAI."""
    if not openai_client:
        return ""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты редактор Telegram-постов. Улучшай тексты: делай их живее, понятнее, убирай канцелярит и воду. Сохраняй смысл и структуру."},
                {"role": "user", "content": f"Улучши этот текст для Telegram-канала. Без пояснений, сразу результат.\n\nТекст:\n{original_text}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("GPT rewrite error:", repr(e))
        return ""


async def rewrite_text_with_ai(original_text: str) -> str:
    return await to_thread(_rewrite_text_sync, original_text)


def _generate_hashtags_sync(post_text: str) -> str:
    """Синхронная генерация хештегов через OpenAI."""
    if not openai_client:
        return ""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты помощник по контенту. Подбираешь релевантные хештеги для Telegram-постов."},
                {"role": "user", "content": f"Подбери 5-10 релевантных хештегов для этого поста. Выведи только хештеги через пробел, без пояснений.\n\nПост:\n{post_text}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("GPT hashtags error:", repr(e))
        return ""


async def generate_hashtags_with_ai(post_text: str) -> str:
    return await to_thread(_generate_hashtags_sync, post_text)


def _generate_variants_sync(post_text: str) -> list:
    """Синхронная генерация A/B вариантов через OpenAI."""
    if not openai_client:
        return []
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты копирайтер. Создаёшь разные варианты одного поста для A/B тестирования."},
                {"role": "user", "content": f"Напиши 3 разных варианта этого поста. Каждый вариант должен отличаться стилем, подачей или акцентами. Раздели варианты строкой '---'. Без пояснений, сразу варианты.\n\nОригинал:\n{post_text}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        variants = [v.strip() for v in text.split("---") if v.strip()]
        return variants
    except Exception as e:
        print("GPT variants error:", repr(e))
        return []


async def generate_variants_with_ai(post_text: str) -> list:
    return await to_thread(_generate_variants_sync, post_text)


def _generate_content_plan_sync(topic: str, period: str) -> str:
    """Синхронная генерация контент-плана через OpenAI."""
    if not openai_client:
        return ""
    period_text = "на неделю (7 постов)" if period == "week" else "на месяц (20-30 постов)"
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты контент-стратег для Telegram-каналов. Создаёшь продуманные контент-планы."},
                {"role": "user", "content": f"Составь контент-план {period_text} для Telegram-канала.\n\nТема канала: {topic}\n\nФормат: пронумерованный список идей постов. Каждая идея — 1-2 предложения. Без пояснений, сразу план."},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("GPT content plan error:", repr(e))
        return ""


async def generate_content_plan_with_ai(topic: str, period: str) -> str:
    return await to_thread(_generate_content_plan_sync, topic, period)


def _copy_style_sync(example_post: str, new_topic: str) -> str:
    """Синхронное копирование стиля через OpenAI."""
    if not openai_client:
        return ""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты копирайтер. Умеешь писать посты в заданном стиле."},
                {"role": "user", "content": f"Напиши новый пост в точно таком же стиле, как пример ниже, но на другую тему.\n\nПример поста (стиль для копирования):\n{example_post}\n\nТема нового поста: {new_topic}\n\nБез пояснений, сразу пост."},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("GPT style copy error:", repr(e))
        return ""


async def copy_style_with_ai(example_post: str, new_topic: str) -> str:
    return await to_thread(_copy_style_sync, example_post, new_topic)


# ----- ШАБЛОНЫ ПОСТОВ -----

POST_TEMPLATES = {
    "news": {
        "name": "📰 Новость",
        "structure": (
            "<b>Структура: Новость</b>\n\n"
            "1. <b>Заголовок</b> — что случилось (1 строка)\n"
            "2. <b>Суть</b> — главное в 2-3 предложениях\n"
            "3. <b>Детали</b> — подробности, цифры, цитаты\n"
            "4. <b>Вывод</b> — почему это важно читателю"
        ),
    },
    "review": {
        "name": "⭐ Обзор",
        "structure": (
            "<b>Структура: Обзор</b>\n\n"
            "1. <b>Заголовок</b> — что обозреваем + оценка/вердикт\n"
            "2. <b>Введение</b> — контекст, зачем это нужно\n"
            "3. <b>Плюсы</b> — что понравилось (список)\n"
            "4. <b>Минусы</b> — что не понравилось (список)\n"
            "5. <b>Вердикт</b> — для кого подойдёт, рекомендация"
        ),
    },
    "story": {
        "name": "📖 История",
        "structure": (
            "<b>Структура: История</b>\n\n"
            "1. <b>Зацепка</b> — интригующее начало\n"
            "2. <b>Контекст</b> — кто, где, когда\n"
            "3. <b>Проблема</b> — с чем столкнулся\n"
            "4. <b>Развитие</b> — что делал, что происходило\n"
            "5. <b>Развязка</b> — чем закончилось\n"
            "6. <b>Урок</b> — что из этого вынести читателю"
        ),
    },
    "tip": {
        "name": "💡 Совет",
        "structure": (
            "<b>Структура: Совет</b>\n\n"
            "1. <b>Заголовок</b> — польза + для кого\n"
            "2. <b>Проблема</b> — с чем сталкиваются люди\n"
            "3. <b>Решение</b> — совет по шагам\n"
            "4. <b>Пример</b> — как это работает\n"
            "5. <b>Призыв</b> — попробуй / поделись опытом"
        ),
    },
    "poll": {
        "name": "📊 Опрос",
        "structure": (
            "<b>Структура: Опрос</b>\n\n"
            "1. <b>Заголовок-вопрос</b> — интересный, провоцирующий\n"
            "2. <b>Контекст</b> — почему спрашиваем (1-2 предложения)\n"
            "3. <b>Варианты</b> — 2-4 варианта ответа\n"
            "4. <b>Призыв</b> — голосуй / пиши в комментариях"
        ),
    },
}


# ----- /rewrite -----


@dp.message(Command("rewrite"))
async def cmd_rewrite(message: types.Message, state: FSMContext):
    """Команда рерайта текста."""
    await state.set_state(RewriteForm.waiting_for_text)
    await message.answer(
        "<b>🔄 Рерайт текста</b>\n\n"
        "Пришли текст поста, который нужно улучшить.\n"
        "ИИ сделает его живее и понятнее.\n\n"
        "Если передумал — /cancel."
    )


@dp.message(RewriteForm.waiting_for_text)
async def process_rewrite(message: types.Message, state: FSMContext):
    """Получаем текст и делаем рерайт."""
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    original_text = (message.text or "").strip()
    if not original_text:
        await message.answer("Пустой текст. Пришли текст поста для рерайта.")
        return

    await message.answer("Улучшаю текст...")

    rewritten = await rewrite_text_with_ai(original_text)

    if not rewritten:
        await state.clear()
        await message.answer("Не удалось улучшить текст. Попробуй ещё раз.")
        return

    await state.update_data(last_generated_post=rewritten, last_generated_idea="Рерайт текста")
    await state.set_state(EditGeneratedPostForm.editing)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="💾 Сохранить", callback_data="genpost_save"),
                InlineKeyboardButton(text="📤 В канал", callback_data="genpost_send"),
            ],
            [InlineKeyboardButton(text="✏️ Редактировать", callback_data="genpost_edit_menu")],
            [InlineKeyboardButton(text="❌ Закрыть", callback_data="genpost_close")],
        ]
    )

    await message.answer(
        f"<b>Улучшенный текст:</b>\n\n{rewritten}",
        reply_markup=kb,
    )


# ----- /hashtags -----


@dp.message(Command("hashtags"))
async def cmd_hashtags(message: types.Message, state: FSMContext):
    """Команда генерации хештегов."""
    await state.set_state(HashtagsForm.waiting_for_text)
    await message.answer(
        "<b>#️⃣ Генерация хештегов</b>\n\n"
        "Пришли текст поста, для которого нужны хештеги.\n\n"
        "Если передумал — /cancel."
    )


@dp.message(HashtagsForm.waiting_for_text)
async def process_hashtags(message: types.Message, state: FSMContext):
    """Получаем текст и генерируем хештеги."""
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    post_text = (message.text or "").strip()
    if not post_text:
        await message.answer("Пустой текст. Пришли текст поста.")
        return

    await message.answer("Подбираю хештеги...")

    hashtags = await generate_hashtags_with_ai(post_text)

    await state.clear()

    if not hashtags:
        await message.answer("Не удалось подобрать хештеги. Попробуй ещё раз.")
        return

    await message.answer(
        f"<b>Хештеги для поста:</b>\n\n{hashtags}\n\n"
        "Скопируй нужные и добавь к посту.",
        reply_markup=main_menu_kb,
    )


# ----- /variants -----


@dp.message(Command("variants"))
async def cmd_variants(message: types.Message, state: FSMContext):
    """Команда генерации A/B вариантов."""
    await state.set_state(VariantsForm.waiting_for_text)
    await message.answer(
        "<b>🎯 A/B варианты</b>\n\n"
        "Пришли текст поста, для которого нужны варианты.\n"
        "ИИ сгенерирует 3 разных версии.\n\n"
        "Если передумал — /cancel."
    )


@dp.message(VariantsForm.waiting_for_text)
async def process_variants(message: types.Message, state: FSMContext):
    """Получаем текст и генерируем варианты."""
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    post_text = (message.text or "").strip()
    if not post_text:
        await message.answer("Пустой текст. Пришли текст поста.")
        return

    await message.answer("Генерирую варианты...")

    variants = await generate_variants_with_ai(post_text)

    await state.clear()

    if not variants:
        await message.answer("Не удалось сгенерировать варианты. Попробуй ещё раз.")
        return

    text = "<b>A/B варианты:</b>\n\n"
    for i, v in enumerate(variants, 1):
        text += f"<b>Вариант {i}:</b>\n{v}\n\n{'─' * 20}\n\n"

    await message.answer(text, reply_markup=main_menu_kb)


# ----- /plan -----


@dp.message(Command("plan"))
async def cmd_plan(message: types.Message, state: FSMContext):
    """Команда генерации контент-плана."""
    await state.set_state(ContentPlanForm.waiting_for_topic)
    await message.answer(
        "<b>📅 Контент-план</b>\n\n"
        "Опиши тему и аудиторию своего канала.\n"
        "Например: «IT-канал для разработчиков, пишем о Python и карьере».\n\n"
        "Если передумал — /cancel."
    )


@dp.message(ContentPlanForm.waiting_for_topic)
async def process_plan_topic(message: types.Message, state: FSMContext):
    """Получаем тему канала, спрашиваем период."""
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    topic = (message.text or "").strip()
    if not topic:
        await message.answer("Пустая тема. Опиши свой канал.")
        return

    await state.update_data(plan_topic=topic)
    await state.set_state(ContentPlanForm.waiting_for_period)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📅 На неделю (7 постов)", callback_data="plan_period:week"),
                InlineKeyboardButton(text="📆 На месяц (20-30 постов)", callback_data="plan_period:month"),
            ],
            [InlineKeyboardButton(text="❌ Отмена", callback_data="plan_cancel")],
        ]
    )

    await message.answer("На какой период сделать план?", reply_markup=kb)


@dp.callback_query(lambda c: c.data and c.data.startswith("plan_period:"))
async def cb_plan_period(callback: types.CallbackQuery, state: FSMContext):
    """Генерируем контент-план."""
    period = callback.data.split(":")[1]
    data = await state.get_data()
    topic = data.get("plan_topic", "")

    await callback.message.answer("Генерирую контент-план...")

    plan = await generate_content_plan_with_ai(topic, period)

    await state.clear()

    if not plan:
        await callback.message.answer("Не удалось сгенерировать план. Попробуй ещё раз.")
        await callback.answer()
        return

    await callback.message.answer(
        f"<b>📅 Контент-план</b>\n\n{plan}",
        reply_markup=main_menu_kb,
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "plan_cancel")
async def cb_plan_cancel(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text("Генерация плана отменена.")
    await callback.answer()


# ----- /templates -----


@dp.message(Command("templates"))
async def cmd_templates(message: types.Message, state: FSMContext):
    """Команда выбора шаблона поста."""
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t["name"], callback_data=f"template:{key}")]
            for key, t in POST_TEMPLATES.items()
        ] + [[InlineKeyboardButton(text="❌ Отмена", callback_data="template_cancel")]]
    )

    await message.answer(
        "<b>📋 Шаблоны постов</b>\n\n"
        "Выбери тип поста, и я покажу структуру для заполнения:",
        reply_markup=kb,
    )


@dp.callback_query(lambda c: c.data and c.data.startswith("template:"))
async def cb_template_select(callback: types.CallbackQuery, state: FSMContext):
    """Показываем структуру выбранного шаблона."""
    template_key = callback.data.split(":")[1]
    template = POST_TEMPLATES.get(template_key)

    if not template:
        await callback.answer("Шаблон не найден.", show_alert=True)
        return

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="← К списку шаблонов", callback_data="template_back")],
        ]
    )

    await callback.message.edit_text(
        f"{template['structure']}\n\n"
        "Используй эту структуру для написания поста.\n"
        "Когда будет готово — сохрани через /draft или «📝 Черновик».",
        reply_markup=kb,
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "template_back")
async def cb_template_back(callback: types.CallbackQuery, state: FSMContext):
    """Возврат к списку шаблонов."""
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t["name"], callback_data=f"template:{key}")]
            for key, t in POST_TEMPLATES.items()
        ] + [[InlineKeyboardButton(text="❌ Отмена", callback_data="template_cancel")]]
    )

    await callback.message.edit_text(
        "<b>📋 Шаблоны постов</b>\n\n"
        "Выбери тип поста, и я покажу структуру для заполнения:",
        reply_markup=kb,
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "template_cancel")
async def cb_template_cancel(callback: types.CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text("Выбор шаблона отменён.")
    await callback.answer()


# ----- /style -----


@dp.message(Command("style"))
async def cmd_style(message: types.Message, state: FSMContext):
    """Команда копирования стиля."""
    await state.set_state(StyleCopyForm.waiting_for_example)
    await message.answer(
        "<b>🎨 Копирование стиля</b>\n\n"
        "Пришли пример поста, стиль которого хочешь скопировать.\n\n"
        "Если передумал — /cancel."
    )


@dp.message(StyleCopyForm.waiting_for_example)
async def process_style_example(message: types.Message, state: FSMContext):
    """Получаем пример поста."""
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    example = (message.text or "").strip()
    if not example:
        await message.answer("Пустой текст. Пришли пример поста.")
        return

    await state.update_data(style_example=example)
    await state.set_state(StyleCopyForm.waiting_for_topic)

    await message.answer(
        "Отлично! Теперь напиши тему нового поста.\n"
        "Например: «5 причин учить Python в 2025».\n\n"
        "Если передумал — /cancel."
    )


@dp.message(StyleCopyForm.waiting_for_topic)
async def process_style_topic(message: types.Message, state: FSMContext):
    """Получаем тему и генерируем пост в скопированном стиле."""
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    new_topic = (message.text or "").strip()
    if not new_topic:
        await message.answer("Пустая тема. Напиши тему нового поста.")
        return

    data = await state.get_data()
    example = data.get("style_example", "")

    await message.answer("Генерирую пост в заданном стиле...")

    new_post = await copy_style_with_ai(example, new_topic)

    if not new_post:
        await state.clear()
        await message.answer("Не удалось сгенерировать пост. Попробуй ещё раз.")
        return

    await state.update_data(last_generated_post=new_post, last_generated_idea=new_topic, attached_media=None)
    await state.set_state(EditGeneratedPostForm.editing)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="💾 Сохранить", callback_data="genpost_save"),
                InlineKeyboardButton(text="📤 В канал", callback_data="genpost_send"),
            ],
            [InlineKeyboardButton(text="✏️ Редактировать", callback_data="genpost_edit_menu")],
            [InlineKeyboardButton(text="❌ Закрыть", callback_data="genpost_close")],
        ]
    )

    await message.answer(
        f"<b>Пост в скопированном стиле:</b>\n\n{new_post}",
        reply_markup=kb,
    )


# ----- /search -----


@dp.message(Command("search"))
async def cmd_search(message: types.Message, state: FSMContext):
    """Поиск по черновикам"""
    await state.set_state(SearchForm.waiting_for_query)
    await message.answer(
        "<b>🔍 Поиск по черновикам</b>\n\n"
        "Введи слово или фразу для поиска.\n\n"
        "Если передумал — /cancel."
    )


# ----- /media_gallery -----


async def show_media_page(message_or_callback, telegram_id: int, page: int = 0, edit: bool = False):
    """Показать медиа-драфты с пагинацией и кнопками просмотра/отправки"""
    rows = await get_user_drafts_full(telegram_id)
    media_rows = []
    for row in rows:
        media_info = parse_media_draft(row.draft_text or "")
        if media_info:
            media_rows.append((row, media_info))

    if not media_rows:
        text = "У тебя пока нет медиа-драфтов. Сохрани через 📎 Медиа."
        if edit and hasattr(message_or_callback, "edit_text"):
            await message_or_callback.edit_text(text)
        else:
            target = message_or_callback.message if hasattr(message_or_callback, "message") else message_or_callback
            await target.answer(text)
        return

    total = len(media_rows)
    total_pages = (total + MEDIA_PER_PAGE - 1) // MEDIA_PER_PAGE
    page = max(0, min(page, total_pages - 1))

    start_idx = page * MEDIA_PER_PAGE
    end_idx = min(start_idx + MEDIA_PER_PAGE, total)
    page_items = media_rows[start_idx:end_idx]

    lines = [f"<b>🖼 Медиатека</b> ({total} шт.)", ""]
    buttons = []

    # Кнопки навигации по страницам
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton(text="← Назад", callback_data=f"media_page:{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="media_page:noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton(text="Вперёд →", callback_data=f"media_page:{page + 1}"))
    buttons.append(nav_buttons)

    # Сами элементы медиатеки + кнопки для каждого
    for i, (row, info) in enumerate(page_items):
        idx = start_idx + i + 1
        caption = (info["caption"] or "—").strip()
        preview = caption[:120] + ("..." if len(caption) > 120 else "")
        lines.append(f"<b>#{idx}</b> {info['type']} — {preview}")

        buttons.append(
            [
                InlineKeyboardButton(text=f"👁 #{idx}", callback_data=f"media_view:{row.id}"),
                InlineKeyboardButton(text=f"📤 #{idx}", callback_data=f"media_send:{row.id}"),
                InlineKeyboardButton(text=f"🗑 #{idx}", callback_data=f"media_del:{row.id}"),
            ]
        )
        lines.append("")

    text = "\n".join(lines).strip()

    # Общие действия
    buttons.append(
        [
            InlineKeyboardButton(text="🔄 Обновить", callback_data=f"media_page:{page}"),
            InlineKeyboardButton(text="📂 Все черновики", callback_data="drafts_page:0"),
        ]
    )

    kb = InlineKeyboardMarkup(inline_keyboard=buttons)

    if edit and hasattr(message_or_callback, "edit_text"):
        await message_or_callback.edit_text(text, reply_markup=kb)
    else:
        target = message_or_callback.message if hasattr(message_or_callback, "message") else message_or_callback
        await target.answer(text, reply_markup=kb)


@dp.message(Command("media"))
async def cmd_media_gallery(message: types.Message, state: FSMContext):
    """Показать медиатеку"""
    user_id = await get_user_id_from_context(message, state)
    await show_media_page(message, user_id, page=0)


@dp.callback_query(lambda c: c.data and c.data.startswith("media_page:"))
async def cb_media_page(callback: types.CallbackQuery, state: FSMContext):
    """Пагинация медиатеки"""
    page_str = callback.data.split(":")[1]
    if page_str == "noop":
        await callback.answer()
        return
    page = int(page_str)
    await state.update_data(_user_telegram_id=callback.from_user.id)
    await show_media_page(callback.message, callback.from_user.id, page=page, edit=True)
    await callback.answer()


@dp.callback_query(lambda c: c.data and c.data.startswith("media_view:"))
async def cb_media_view(callback: types.CallbackQuery, state: FSMContext):
    """Показать медиа пользователю"""
    try:
        draft_id = int(callback.data.split(":")[1])
    except Exception:
        await callback.answer("Не понял, что показать.", show_alert=True)
        return

    user_id = callback.from_user.id
    draft = await get_user_draft_by_id(user_id, draft_id)
    if not draft:
        await callback.answer("Черновик не найден.", show_alert=True)
        return

    media_info = parse_media_draft(draft.draft_text or "")
    if not media_info:
        await callback.answer("Это не медиа-драфт.", show_alert=True)
        return

    caption = media_info["caption"] or None
    mtype = media_info["type"]
    fid = media_info["file_id"]

    try:
        if mtype == "photo":
            await bot.send_photo(chat_id=callback.from_user.id, photo=fid, caption=caption)
        elif mtype == "video":
            await bot.send_video(chat_id=callback.from_user.id, video=fid, caption=caption)
        elif mtype == "video_note":
            await bot.send_video_note(chat_id=callback.from_user.id, video_note=fid)
            if caption:
                await bot.send_message(chat_id=callback.from_user.id, text=caption)
        elif mtype == "document":
            await bot.send_document(chat_id=callback.from_user.id, document=fid, caption=caption)
        elif mtype == "voice":
            await bot.send_voice(chat_id=callback.from_user.id, voice=fid, caption=caption)
        else:
            await bot.send_message(chat_id=callback.from_user.id, text=caption or "Медиа без подписи")
    except Exception as e:
        await callback.answer(f"Не удалось отправить медиа: {e}", show_alert=True)
        return

    await callback.answer("Готово.")


@dp.callback_query(lambda c: c.data and c.data.startswith("media_send:"))
async def cb_media_send(callback: types.CallbackQuery, state: FSMContext):
    """Начать отправку медиа-драфта в канал"""
    try:
        draft_id = int(callback.data.split(":")[1])
    except Exception:
        await callback.answer("Не понял, что отправлять.", show_alert=True)
        return

    user_id = callback.from_user.id
    draft = await get_user_draft_by_id(user_id, draft_id)
    if not draft:
        await callback.answer("Черновик не найден.", show_alert=True)
        return

    # Сохраняем медиа-драфт в state и переходим к запросу канала
    await state.update_data(draft_text=draft.draft_text, draft_number=f"media-{draft_id}", draft_id=draft_id, _user_telegram_id=user_id)
    await state.set_state(SendDraftForm.waiting_for_channel)

    await callback.message.answer(
        "Медиа-драфт выбран. Пришли @username канала или chat_id, куда отправить.\n"
        "Пример: @mychannel или -1001234567890.\n\n"
        "Если передумал — /cancel."
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data and c.data.startswith("media_del:"))
async def cb_media_del(callback: types.CallbackQuery, state: FSMContext):
    """Удалить медиа-драфт"""
    try:
        draft_id = int(callback.data.split(":")[1])
    except Exception:
        await callback.answer("Не понял, что удалить.", show_alert=True)
        return

    user_id = callback.from_user.id
    ok = await delete_user_draft(user_id, draft_id)
    if ok:
        await callback.answer("Удалено.")
        await show_media_page(callback.message, user_id, page=0, edit=True)
    else:
        await callback.answer("Не найдено или уже удалено.", show_alert=True)


@dp.message(SearchForm.waiting_for_query)
async def process_search(message: types.Message, state: FSMContext):
    """Выполнить поиск"""
    if (message.text or "").strip().lower() == "/cancel":
        return await cmd_cancel(message, state)

    query = (message.text or "").strip().lower()
    if not query:
        await message.answer("Пустой запрос. Введи слово для поиска.")
        return

    user_id = await get_user_id_from_context(message, state)
    rows = await get_user_drafts_full(user_id)

    if not rows:
        await state.clear()
        await message.answer("У тебя пока нет черновиков для поиска.", reply_markup=main_menu_kb)
        return

    # Ищем совпадения
    results = []
    for idx, row in enumerate(rows, start=1):
        draft_text = (row.draft_text or "").lower()
        idea_text = (row.idea_text or "").lower()
        if query in draft_text or query in idea_text:
            results.append((idx, row))

    await state.clear()

    if not results:
        await message.answer(
            f"По запросу «{query}» ничего не найдено.\n\n"
            "Попробуй другой запрос или посмотри все черновики: /my_drafts",
            reply_markup=main_menu_kb,
        )
        return

    lines = [f"<b>🔍 Результаты поиска</b> «{query}»", f"Найдено: {len(results)}", ""]

    for idx, row in results[:10]:  # Показываем максимум 10
        draft_text = (row.draft_text or "").strip()
        media_info = parse_media_draft(draft_text)

        if media_info:
            preview = f"📎 {media_info['type']}: {(media_info['caption'] or '—')[:80]}..."
        else:
            preview = draft_text[:120] + ("..." if len(draft_text) > 120 else "")

        lines.append(f"<b>#{idx}</b> {preview}")
        lines.append("")

    if len(results) > 10:
        lines.append(f"<i>...и ещё {len(results) - 10} результатов</i>")

    text = "\n".join(lines).strip()

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✏️ Редактировать", callback_data="quick:edit"),
                InlineKeyboardButton(text="🗑 Удалить", callback_data="quick:delete"),
            ],
            [
                InlineKeyboardButton(text="📤 Отправить", callback_data="quick:send"),
                InlineKeyboardButton(text="📂 Все черновики", callback_data="drafts_page:0"),
            ],
        ]
    )

    await message.answer(text, reply_markup=kb)


# ---------- ТОЧКА ВХОДА ----------

async def main():
    global session_factory
    session_factory = SessionLocal

    await init_db()
    print("Бот запущен. Нажми Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())