from aiogram import types
from aiogram.dispatcher.filters import Text
from bot_parts.reply import start_text, discription_SRGAN, bot_functional, start_transformation, bot_info
from bot_parts.keyboards import start_keyboard, kb_cancel, create_result, get_back_to_main_menu
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher import FSMContext
import os
import asyncio
from bot_init import dp, bot, storage


class ClientStatesGroup(StatesGroup):
    load_photo = State()
    confirm_run = State()

#  Стартовая команда для начала взаимодействия с ботом
@dp.message_handler(commands=['start'])
async def start_command(messege: types.Message):
    await messege.answer(text=start_text,
                         parse_mode='HTML',
                         reply_markup=start_keyboard)
    await messege.delete()


#  Функция отмены обработки фотографии
@dp.message_handler(commands=['cancel'], state='*')
async def cmd_cancel(message: types.Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state is None:
        return

    await message.reply('Canceled',
                        reply_markup=start_keyboard)
    await state.finish()


# Функция возвращения в главное меню из любого раздела бота
@dp.message_handler(Text(equals='🔙 Back to main menu'))
async def main_menu_command(messege: types.Message):
    await messege.answer(text='Select the option ',
                         parse_mode='HTML',
                         reply_markup=start_keyboard)
    await messege.delete()


#  Функция перемещения в раздел, описывающий функционал бота
@dp.message_handler(Text(equals='🔧 Bot functional'))
async def bot_functional_command(messege: types.Message):
    await messege.answer(text=bot_functional,
                         parse_mode='HTML',
                         reply_markup=get_back_to_main_menu())
    await messege.delete()


#  Функция описывающая и показывающая работу нейронной сети
@dp.message_handler(Text(equals='🖊 Neural Network description'))
async def discription_sgran(messege: types.Message):
    await bot.send_photo(chat_id=messege.from_user.id,
                         photo='https://miro.medium.com/max/1035/1*bRYFXzsureiJjsSE9ZpzNw.png')
    await messege.answer(text=discription_SRGAN,
                         parse_mode='HTML',
                         reply_markup=get_back_to_main_menu())
    await messege.delete()


#  Функция описывающая возможные команды бота
@dp.message_handler(Text(equals='📜 Bot info'))
async def bot_info_command(messege: types.Message):
    await messege.answer(text=bot_info,
                         parse_mode='HTML',
                         reply_markup=get_back_to_main_menu())
    await messege.delete()


#  Функция направляющая в раздел трансформации исходной фотографии
@dp.message_handler(Text(equals='⚡️ Start transformation', ignore_case=True), state=None)
async def start_work(message: types.Message) -> None:
    await ClientStatesGroup.load_photo.set()
    await message.answer(text=start_transformation,
                         reply_markup=kb_cancel)


# Функция обработки фотографии
@dp.message_handler(Text(equals='Create result', ignore_case=True), state=ClientStatesGroup.load_photo)
async def start_process(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        id_photo = data['photo']
        photo_path = f"photos/{id_photo}.jpg"
    save_path = f"result/{id_photo}.jpg"
    args = f'python nn_model/model.py {photo_path} {save_path}'.split()
    p = await asyncio.create_subprocess_exec(*args)
    await p.wait()

    result = types.InputFile(save_path)

    await message.answer_photo(result,
                               reply_markup=start_keyboard)
    await clear_files(state)
    await state.finish()

#  Проверка сообщения на наличие фотографии
@dp.message_handler(lambda message: not message.photo, state=ClientStatesGroup.load_photo)
async def check_photo(message: types.Message):
    return await message.reply(text='its not a photo!',
                               reply_markup=kb_cancel)


#  Функция загрузки фотографии
@dp.message_handler(lambda message: message.photo, content_types=['photo'], state=ClientStatesGroup.load_photo)
async def load_photo(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['photo'] = message.photo[-1].file_id

    await message.photo[-1].download(destination_file=f'photos/{message.photo[-1].file_id}.jpg')
    # await ClientStatesGroup.next()
    await message.reply('Just one second, you can stop process',
                        reply_markup=create_result)


#  Функция очистки фотографий после отправки обработанного фото
async def clear_files(state: FSMContext):
    async with state.proxy() as data:
        if 'photo' in data:
            photo_id = data['photo']
            os.system(f'rm photos/{photo_id}*')
            os.system(f'rm result/{photo_id}*')
