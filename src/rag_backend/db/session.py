"""SQLAlchemy engine and request-scoped database sessions."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_backend.core.config import settings

engine = create_async_engine(settings.database_url, pool_pre_ping=True)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Provide one database session and close it after the request."""
    async with AsyncSessionFactory() as session:
        yield session

