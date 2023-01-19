from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

# Клавиатура для быстрого взаимодействия с основным интерфейсом бота
start_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)

start_botton1 = KeyboardButton('🔧 Bot functional')
start_botton2 = KeyboardButton('🖊 Neural Network description')
start_botton3 = KeyboardButton('⚡️ Start transformation')
start_botton4 = KeyboardButton('📜 Bot info')
start_keyboard.add(start_botton1).add(start_botton2).add(start_botton3).add(start_botton4)


#  Клавиатура для возвращения в главное меню
def get_back_to_main_menu():
    bot_functional_keyboard = ReplyKeyboardMarkup(resize_keyboard=True)

    bot_functional_botton1 = KeyboardButton('🔙 Back to main menu')
    bot_functional_keyboard.add(bot_functional_botton1)
    return bot_functional_keyboard


kb_cancel = ReplyKeyboardMarkup(resize_keyboard=True)
kb_cancel.add(KeyboardButton('/cancel'))


create_result = ReplyKeyboardMarkup(resize_keyboard=True)
create_result.add(KeyboardButton('Create result')).add(KeyboardButton('/cancel'))
