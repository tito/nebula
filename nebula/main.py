from nebula.bot import main as main_bot
from nebula.web import main as main_web
import asyncio


async def main():
    await asyncio.gather(main_bot(), main_web())


if __name__ == "__main__":
    asyncio.run(main())
