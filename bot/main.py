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


# ---------- КЛАВИАТУРА ----------

main_menu_kb = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="/help"),
            KeyboardButton(text="✨ Идеи постов"),
            KeyboardButton(text="📝 Новый черновик"),
        ],
        [
            KeyboardButton(text="📂 Мои черновики"),
            KeyboardButton(text="🗑 Удалить черновик"),
            KeyboardButton(text="✏️ Редактировать черновик"),
            KeyboardButton(text="📤 Отправить черновик"),
            KeyboardButton(text="📎 Сохранить медиа"),
        ],
    ],
    resize_keyboard=True,
)


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
    await message.answer(
        "Привет! Я ИИ‑ассистент для планирования контента ТГ‑канала.\n"
        "Можешь пользоваться кнопками ниже или командами.",
        reply_markup=main_menu_kb,
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "/start - начать работу\n"
        "/help - список команд\n"
        "/idea - сгенерировать идеи постов\n"
        "/draft - создать черновик поста по своей идее\n"
        "/my_drafts - показать сохранённые черновики\n"
        "/delete_draft - удалить один из сохранённых черновиков\n"
        "/edit_draft - отредактировать сохранённый черновик\n"
        "/send_draft - отправить выбранный черновик в канал\n"
        "/cancel - отменить текущий диалог\n\n"
        "Или используй кнопки ниже 👇",
        reply_markup=main_menu_kb,
    )


# ----- КНОПКИ (Reply-клавиатура) -----


@dp.message(lambda m: m.text == "✨ Идеи постов")
async def btn_ideas(message: types.Message, state: FSMContext):
    await cmd_idea(message, state)


@dp.message(lambda m: m.text == "📝 Новый черновик")
async def btn_new_draft(message: types.Message, state: FSMContext):
    await cmd_draft(message, state)


@dp.message(lambda m: m.text == "📂 Мои черновики")
async def btn_my_drafts(message: types.Message):
    await cmd_my_drafts(message)


@dp.message(lambda m: m.text == "🗑 Удалить черновик")
async def btn_delete_draft(message: types.Message, state: FSMContext):
    await cmd_delete_draft(message, state)


@dp.message(lambda m: m.text == "✏️ Редактировать черновик")
async def btn_edit_draft(message: types.Message, state: FSMContext):
    await cmd_edit_draft(message, state)


@dp.message(lambda m: m.text == "📤 Отправить черновик")
async def btn_send_draft(message: types.Message, state: FSMContext):
    await cmd_send_draft(message, state)


@dp.message(lambda m: m.text == "📎 Сохранить медиа")
async def btn_save_media(message: types.Message, state: FSMContext):
    await cmd_save_media_draft(message, state)


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
    if not text.isdigit():
        await message.answer("Номер должен быть числом. Пришли, пожалуйста, номер черновика (например: 2).")
        return

    draft_number = int(text)
    drafts = await get_user_drafts_full(message.from_user.id)

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
    """
    channel = (message.text or "").strip()
    data = await state.get_data()
    draft_text = data.get("draft_text", "")
    draft_number = data.get("draft_number")

    if not draft_text:
        await state.clear()
        await message.answer("Не получилось получить текст черновика. Попробуй ещё раз /send_draft.")
        return

    await state.clear()

    # Если это медиа-драфт (храним как MEDIA|type|file_id|caption)
    media_info = parse_media_draft(draft_text)

    try:
        if media_info:
            mtype = media_info["type"]
            fid = media_info["file_id"]
            caption = media_info["caption"] or None

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
                # fallback на текст
                await bot.send_message(chat_id=channel, text=draft_text)
        else:
            await bot.send_message(chat_id=channel, text=draft_text)

        await message.answer(
            f"Черновик №{draft_number} отправлен в канал {channel}.\n"
            "Не забудь, что у бота должны быть права на публикацию."
        )
    except TelegramForbiddenError:
        await message.answer(
            "Не удалось отправить сообщение: похоже, у бота нет прав в этом канале.\n"
            "Убедись, что бот — админ, и попробуй снова."
        )
    except Exception as e:
        await message.answer(
            "Не получилось отправить сообщение в канал. "
            "Проверь @username / chat_id и попробуй ещё раз."
        )


# Старт отправки из инлайн-кнопки
@dp.callback_query(lambda c: c.data == "start_send_draft")
async def cb_start_send_draft(callback: types.CallbackQuery, state: FSMContext):
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

    if not text.isdigit():
        await message.answer("Номер должен быть числом. Пришли, пожалуйста, номер черновика (например: 2).")
        return

    draft_number = int(text)
    drafts = await get_user_drafts_full(message.from_user.id)

    if draft_number < 1 or draft_number > len(drafts):
        await message.answer(
            "Черновик с таким номером не найден среди твоих.\n"
            "Проверь номер в /my_drafts и попробуй ещё раз, или напиши /cancel."
        )
        return

    draft = drafts[draft_number - 1]
    await state.update_data(draft_id=draft.id, draft_number=draft_number)

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

    await create_draft(
        telegram_id=message.from_user.id,
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

    draft_id = await create_draft(
        telegram_id=message.from_user.id,
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
        "- Стиль: живой, понятный широкой аудитории, без канцелярита.\n"
        "- Структура: цепляющее вступление, 2–4 абзаца основной части, короткое заключение с выводом/призывом.\n"
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

    # Сохраняем сгенерированный пост в состоянии, чтобы по кнопке можно было сохранить в черновики
    await state.update_data(
        last_generated_idea=idea_text,
        last_generated_post=post_text,
    )
    await state.set_state(PlanForm.generated_post)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="💾 Сохранить в черновики",
                    callback_data="save_generated_post",
                )
            ],
            [InlineKeyboardButton(text="❌ Закрыть", callback_data="close_generated_post")],
        ]
    )

    await callback.message.answer(
        f"<b>Готовый пост по идее:</b>\n{post_text}\n\n"
        "Сохранить в черновики?",
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


# ----- /my_drafts -----

@dp.message(Command("my_drafts"))
async def cmd_my_drafts(message: types.Message):
    """
    Показываем все черновики пользователя с порядковыми номерами (1,2,3...) по времени создания.
    Номера используются для удаления/редактирования.
    """
    rows = await get_user_drafts_full(message.from_user.id)

    if not rows:
        await message.answer("У тебя пока нет сохранённых черновиков.")
        return

    parts = []
    for idx, row in enumerate(rows, start=1):
        idea_text = (row.idea_text or "").strip()
        draft_text = (row.draft_text or "").strip()

        media_info = parse_media_draft(draft_text)

        if media_info:
            mtype = media_info["type"]
            caption = media_info["caption"]
            parts.append(
                f"<b>Черновик {idx}</b>\n"
                f"Медиа: {mtype}\n"
                f"Подпись:\n{caption or '—'}\n"
                "────────────"
            )
        else:
            parts.append(
                f"<b>Черновик {idx}</b>\n"
                f"{draft_text}\n"
                "────────────"
            )

    text = "Твои черновики:\n\n" + "\n".join(parts)
    await message.answer(text)


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

    if not text.isdigit():
        await message.answer("Номер должен быть числом. Пришли, пожалуйста, номер черновика (например: 2).")
        return

    draft_number = int(text)
    drafts = await get_user_drafts_full(message.from_user.id)

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


# ---------- ТОЧКА ВХОДА ----------

async def main():
    global session_factory
    session_factory = SessionLocal

    await init_db()
    print("Бот запущен. Нажми Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())