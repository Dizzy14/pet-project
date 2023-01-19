from aiogram import executor
import asyncio
from bot_init import dp
from bot_parts import handlers, keyboards, reply


async def on_startup(_):
    print('lets go')


async def on_shutdown(_):
    await asyncio.create_subprocess_shell('rm photos/* result/*')



if __name__ == '__main__':
    executor.start_polling(dp,
                           skip_updates=True,
                           on_startup=on_startup,
                           on_shutdown=on_shutdown)
