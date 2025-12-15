import os
from datetime import datetime
from typing import AsyncIterator, List

from dotenv import load_dotenv
from sqlalchemy import BigInteger, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "tg_content_db")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

DATABASE_URL = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    drafts: Mapped[List["Draft"]] = relationship("Draft", back_populates="user")


class Draft(Base):
    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    idea_text: Mapped[str] = mapped_column(Text, nullable=False)
    draft_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped[User] = relationship("User", back_populates="drafts")


engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db():
    """
    Создаём таблицы, если их ещё нет. Для продакшена лучше использовать миграции (Alembic).
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> AsyncIterator[AsyncSession]:
    """
    Удобный фабричный вызов для получения сессии.
    Использование: async with get_session() as session: ...
    """
    return SessionLocal()