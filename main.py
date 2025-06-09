import os
import io
import zipfile
import base64
import asyncio
import logging
from typing import List, Tuple
from datetime import datetime

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from openai import AsyncOpenAI
from PIL import Image
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация клиентов
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Константы
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}

class ImageAnalyzer:
    def __init__(self):
        self.openai_client = openai_client

    async def analyze_image(self, image_data: bytes, filename: str) -> str:
        """Анализирует изображение с помощью OpenAI Vision API по заданным критериям"""
        try:
            # Конвертируем изображение в base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Проанализируйте изображение по следующим критериям и дайте краткие ответы на русском языке:

а. Есть ли на картинке реалистичное фото? (да/нет)
б. Есть ли на картинке иллюстрация? (да/нет) 
в. Что изображено на картинке крупнее всего: люди или какие именно предметы?
г. Каков основной цвет фона?
д. Содержится ли на картинке сообщение о скидке или выгоде? (да/нет)

Ответьте строго по формату:
а. [ответ]
б. [ответ]
в. [ответ]
г. [ответ]
д. [ответ]"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при анализе изображения {filename}: {e}")
            return f"Ошибка при анализе изображения: {str(e)}"

    def is_valid_image(self, data: bytes) -> bool:
        """Проверяет, является ли файл валидным изображением"""
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
            return True
        except Exception:
            return False

class TelegramBot:
    def __init__(self):
        self.analyzer = ImageAnalyzer()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = """
🤖 *Бот для анализа изображений*

Привет! Я анализирую изображения из ZIP архивов по конкретным критериям.

📋 *Критерии анализа:*
а. Есть ли реалистичное фото?
б. Есть ли иллюстрация?
в. Что изображено крупнее всего?
г. Основной цвет фона
д. Есть ли сообщение о скидке?

📝 *Как использовать:*
1. Отправьте ZIP архив с изображениями (JPG/PNG)
2. Получите структурированный анализ каждого изображения

⚠️ *Ограничения:*
• Максимальный размер файла: 20MB
• Поддерживаемые форматы: JPG, JPEG, PNG
• Максимум 10 изображений за раз

Просто отправьте ZIP файл для начала анализа!
        """
        
        await update.message.reply_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN
        )

    async def extract_images_from_zip(self, zip_data: bytes) -> List[Tuple[str, bytes]]:
        """Извлекает изображения из ZIP архива"""
        images = []
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
                for file_info in zip_file.filelist:
                    # Пропускаем директории
                    if file_info.is_dir():
                        continue
                    
                    filename = file_info.filename
                    file_ext = os.path.splitext(filename.lower())[1]
                    
                    # Проверяем формат файла
                    if file_ext not in SUPPORTED_FORMATS:
                        continue
                    
                    # Извлекаем данные файла
                    try:
                        file_data = zip_file.read(file_info)
                        
                        # Проверяем, что это валидное изображение
                        if self.analyzer.is_valid_image(file_data):
                            images.append((filename, file_data))
                            
                            # Ограничиваем количество изображений
                            if len(images) >= 10:
                                break
                                
                    except Exception as e:
                        logger.warning(f"Не удалось извлечь файл {filename}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Ошибка при извлечении ZIP архива: {e}")
            raise Exception(f"Ошибка при обработке ZIP архива: {str(e)}")
        
        return images

    async def process_zip_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает ZIP файл с изображениями"""
        try:
            # Проверяем тип файла
            if not update.message.document:
                await update.message.reply_text("❌ Пожалуйста, отправьте ZIP файл.")
                return
            
            document = update.message.document
            
            # Проверяем расширение файла
            if not document.file_name.lower().endswith('.zip'):
                await update.message.reply_text("❌ Пожалуйста, отправьте файл с расширением .zip")
                return
            
            # Проверяем размер файла
            if document.file_size > MAX_FILE_SIZE:
                await update.message.reply_text(f"❌ Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE // (1024*1024)}MB")
                return
            
            # Отправляем сообщение о начале обработки
            processing_message = await update.message.reply_text("🔄 Обрабатываю ZIP архив...")
            
            # Скачиваем файл
            file = await context.bot.get_file(document.file_id)
            zip_data = await file.download_as_bytearray()
            
            # Извлекаем изображения
            await processing_message.edit_text("📂 Извлекаю изображения из архива...")
            images = await self.extract_images_from_zip(bytes(zip_data))
            
            if not images:
                await processing_message.edit_text("❌ В архиве не найдено валидных изображений (JPG/PNG).")
                return
            
            await processing_message.edit_text(f"🖼️ Найдено {len(images)} изображений. Анализирую содержимое...")
            
            # Анализируем изображения
            results = []
            for i, (filename, image_data) in enumerate(images, 1):
                await processing_message.edit_text(f"🔍 Анализирую изображение {i}/{len(images)}: {filename}")
                
                description = await self.analyzer.analyze_image(image_data, filename)
                results.append((filename, description))
                
                # Небольшая задержка для избежания rate limit
                await asyncio.sleep(0.5)
            
            # Форматируем результаты в таблицу
            await processing_message.edit_text("📊 Формирую результаты...")
            table = self.format_results_table(results)
            
            # Отправляем результаты
            await processing_message.delete()
            
            # Разбиваем длинное сообщение на части если нужно
            if len(table) > 4096:
                parts = self.split_message(table, 4096)
                for i, part in enumerate(parts):
                    await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(table, parse_mode=ParseMode.MARKDOWN)
                
        except Exception as e:
            logger.error(f"Ошибка при обработке ZIP файла: {e}")
            await update.message.reply_text(f"❌ Произошла ошибка при обработке файла: {str(e)}")

    def format_results_table(self, results: List[Tuple[str, str]]) -> str:
        """Форматирует результаты в виде таблицы"""
        if not results:
            return "❌ Нет результатов для отображения."
        
        table = "📊 *Результаты анализа изображений*\n\n"
        table += f"🕒 Время обработки: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
        table += f"📁 Обработано файлов: {len(results)}\n\n"
        
        for i, (filename, description) in enumerate(results, 1):
            table += f"*{i}\\. {self.escape_markdown(filename)}*\n"
            
            # Форматируем структурированный анализ
            lines = description.split('\n')
            formatted_analysis = ""
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Ошибка'):
                    if line.startswith(('а.', 'б.', 'в.', 'г.', 'д.')):
                        formatted_analysis += f"  {self.escape_markdown(line)}\n"
                    else:
                        formatted_analysis += f"{self.escape_markdown(line)}\n"
            
            if formatted_analysis.strip():
                table += f"📋 *Анализ:*\n{formatted_analysis}\n"
            else:
                table += f"📝 {self.escape_markdown(description)}\n\n"
            
            table += "─" * 40 + "\n\n"
        
        return table

    def escape_markdown(self, text: str) -> str:
        """Экранирует специальные символы для Markdown"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    def split_message(self, text: str, max_length: int) -> List[str]:
        """Разбивает длинное сообщение на части"""
        parts = []
        current_part = ""
        
        lines = text.split('\n')
        for line in lines:
            if len(current_part + line + '\n') > max_length:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = line + '\n'
                else:
                    # Если одна строка слишком длинная, разбиваем её
                    while len(line) > max_length:
                        parts.append(line[:max_length])
                        line = line[max_length:]
                    current_part = line + '\n'
            else:
                current_part += line + '\n'
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts

def main():
    """Главная функция запуска бота"""
    # Проверяем наличие необходимых токенов
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN не найден в переменных окружения")
        return
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY не найден в переменных окружения")
        return
    
    # Создаем экземпляр бота
    bot = TelegramBot()
    
    # Создаем приложение
    application = Application.builder().token(telegram_token).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(MessageHandler(filters.Document.ZIP, bot.process_zip_file))
    
    # Обработчик для всех остальных сообщений
    async def handle_other_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "❌ Пожалуйста, отправьте ZIP архив с изображениями или используйте команду /start для получения инструкций."
        )
    
    application.add_handler(MessageHandler(filters.ALL & ~filters.Document.ZIP, handle_other_messages))
    
    # Запускаем бота
    logger.info("Запускаем бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 