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
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif'}
MAX_GIF_FRAMES = 5  # Максимум кадров для анализа в GIF

class ImageAnalyzer:
    def __init__(self):
        self.openai_client = openai_client

    async def analyze_image(self, image_data: bytes, filename: str) -> str:
        """Анализирует изображение с помощью OpenAI Vision API по заданным критериям"""
        try:
            # Проверяем, является ли файл GIF
            if filename.lower().endswith('.gif'):
                frames = self.extract_gif_frames(image_data)
                if frames:
                    return await self.analyze_gif_frames(frames, filename)
                else:
                    return "Ошибка: не удалось извлечь кадры из GIF файла"
            
            # Для обычных изображений (JPG, PNG)
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

    def extract_gif_frames(self, gif_data: bytes) -> List[bytes]:
        """Извлекает ключевые кадры из GIF анимации"""
        frames = []
        try:
            with Image.open(io.BytesIO(gif_data)) as gif:
                if not getattr(gif, 'is_animated', False):
                    # Если это не анимированный GIF, обрабатываем как обычное изображение
                    gif_copy = gif.copy()
                    frame_buffer = io.BytesIO()
                    gif_copy.save(frame_buffer, format='PNG')
                    frames.append(frame_buffer.getvalue())
                    return frames
                
                frame_count = getattr(gif, 'n_frames', 1)
                
                # Выбираем кадры равномерно по всей анимации
                step = max(1, frame_count // MAX_GIF_FRAMES)
                selected_frames = list(range(0, frame_count, step))[:MAX_GIF_FRAMES]
                
                for frame_idx in selected_frames:
                    gif.seek(frame_idx)
                    # Конвертируем в RGB если нужно
                    frame = gif.convert('RGB')
                    
                    # Сохраняем кадр в PNG формате
                    frame_buffer = io.BytesIO()
                    frame.save(frame_buffer, format='PNG')
                    frames.append(frame_buffer.getvalue())
                    
        except Exception as e:
            logger.warning(f"Ошибка при извлечении кадров GIF: {e}")
            
        return frames

    async def analyze_gif_frames(self, frames: List[bytes], filename: str) -> str:
        """Анализирует кадры GIF с игнорированием дисклеймеров"""
        if not frames:
            return "Ошибка: не удалось извлечь кадры из GIF"
        
        try:
            # Анализируем каждый кадр
            frame_analyses = []
            
            for i, frame_data in enumerate(frames):
                base64_image = base64.b64encode(frame_data).decode('utf-8')
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Проанализируйте кадр {i+1} из GIF анимации по следующим критериям. ВАЖНО: игнорируйте любые дисклеймеры, юридические уведомления, мелкий текст с правовой информацией, предупреждения о рисках.

а. Есть ли на картинке реалистичное фото? (да/нет)
б. Есть ли на картинке иллюстрация? (да/нет) 
в. Что изображено на картинке крупнее всего: люди или какие именно предметы?
г. Каков основной цвет фона?
д. Содержится ли на картинке сообщение о скидке или выгоде? (да/нет, игнорируйте юридические дисклеймеры)

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
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=200
                )
                
                frame_analysis = response.choices[0].message.content.strip()
                frame_analyses.append(f"Кадр {i+1}:\n{frame_analysis}")
                
                # Небольшая задержка между запросами
                await asyncio.sleep(0.3)
            
            # Объединяем анализ всех кадров и делаем общий вывод
            combined_analysis = await self.combine_frame_analyses(frame_analyses, filename)
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Ошибка при анализе GIF {filename}: {e}")
            return f"Ошибка при анализе GIF: {str(e)}"

    async def combine_frame_analyses(self, frame_analyses: List[str], filename: str) -> str:
        """Объединяет анализ кадров в общий результат"""
        try:
            combined_text = "\n\n".join(frame_analyses)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""На основе анализа {len(frame_analyses)} кадров GIF анимации, дайте ОБЩИЙ итоговый анализ по тем же критериям. Учитывайте доминирующие характеристики по всем кадрам:

{combined_text}

Дайте итоговый ответ по формату:
а. [общий ответ по всем кадрам]
б. [общий ответ по всем кадрам]
в. [что доминирует по всем кадрам]
г. [преобладающий цвет фона]
д. [есть ли промо-контент, игнорируя дисклеймеры]"""
                    }
                ],
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Ошибка при объединении анализа кадров для {filename}: {e}")
            # Возвращаем анализ первого кадра как fallback
            if frame_analyses:
                return frame_analyses[0]
            return "Ошибка при объединении анализа кадров"

    def parse_analysis_results(self, analysis_text: str) -> dict:
        """Парсит результаты анализа в структурированный формат"""
        results = {
            'realistic_photo': 'unknown',
            'illustration': 'unknown', 
            'main_object': 'unknown',
            'background_color': 'unknown',
            'discount_message': 'unknown'
        }
        
        lines = analysis_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('а.'):
                results['realistic_photo'] = 'yes' if 'да' in line.lower() else 'no'
            elif line.startswith('б.'):
                results['illustration'] = 'yes' if 'да' in line.lower() else 'no'
            elif line.startswith('в.'):
                # Извлекаем основной объект после "в."
                obj_text = line[2:].strip().lower()
                if 'люди' in obj_text or 'человек' in obj_text:
                    results['main_object'] = 'people'
                elif any(word in obj_text for word in ['телефон', 'компьютер', 'машина', 'автомобиль']):
                    results['main_object'] = 'tech'
                elif any(word in obj_text for word in ['еда', 'продукт', 'товар', 'одежда']):
                    results['main_object'] = 'product'
                else:
                    # Берем первое слово как основной объект
                    words = obj_text.split()
                    if words:
                        results['main_object'] = words[0][:10]  # Ограничиваем длину
            elif line.startswith('г.'):
                color_text = line[2:].strip().lower()
                colors_map = {
                    'белый': 'white', 'черный': 'black', 'красный': 'red',
                    'синий': 'blue', 'зеленый': 'green', 'желтый': 'yellow',
                    'серый': 'gray', 'коричневый': 'brown', 'розовый': 'pink'
                }
                for ru_color, en_color in colors_map.items():
                    if ru_color in color_text:
                        results['background_color'] = en_color
                        break
                else:
                    # Если не нашли стандартный цвет, берем первое слово
                    words = color_text.split()
                    if words:
                        results['background_color'] = words[0][:8]
            elif line.startswith('д.'):
                results['discount_message'] = 'yes' if 'да' in line.lower() else 'no'
        
        return results

    def create_new_filename(self, original_filename: str, analysis_results: dict) -> str:
        """Создает новое имя файла на основе результатов анализа"""
        # Получаем расширение файла
        name, ext = os.path.splitext(original_filename)
        
        # Создаем компактную схему именования
        # Формат: [GIF-]R[0/1]-I[0/1]-[obj]-[color]-S[0/1]_original
        photo = '1' if analysis_results['realistic_photo'] == 'yes' else '0'
        illus = '1' if analysis_results['illustration'] == 'yes' else '0'
        obj = analysis_results['main_object']
        color = analysis_results['background_color']
        sale = '1' if analysis_results['discount_message'] == 'yes' else '0'
        
        # Очищаем имя от специальных символов
        clean_name = "".join(c for c in name if c.isalnum() or c in ('-', '_'))[:20]
        
        # Добавляем префикс для GIF файлов
        gif_prefix = "GIF-" if ext.lower() == '.gif' else ""
        
        new_name = f"{gif_prefix}R{photo}-I{illus}-{obj}-{color}-S{sale}_{clean_name}{ext}"
        
        return new_name

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

📝 *Что получите:*
1. Структурированный анализ каждого изображения
2. Переименованный ZIP архив с новыми названиями файлов
3. README файл с расшифровкой схемы именования

🚀 *Схема именования:*
`R1-I0-people-blue-S0_original.jpg`
`GIF-R0-I1-tech-white-S1_banner.gif`
• R1 = реалистичное фото
• I0 = не иллюстрация  
• people = основной объект
• blue = цвет фона
• S0 = нет скидки
• GIF- = префикс для анимации

🎞️ *GIF анимации:*
• Анализ по ключевым кадрам (макс. 5)
• Игнорирование юридических дисклеймеров
• Общий анализ по всей анимации

⚠️ *Ограничения:*
• Максимальный размер файла: 20MB
• Поддерживаемые форматы: JPG, JPEG, PNG, GIF
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
            images_with_analysis = []  # Для создания переименованного архива
            
            for i, (filename, image_data) in enumerate(images, 1):
                if filename.lower().endswith('.gif'):
                    await processing_message.edit_text(f"🎞️ Анализирую GIF анимацию {i}/{len(images)}: {filename}\n(анализ кадров может занять больше времени)")
                else:
                    await processing_message.edit_text(f"🔍 Анализирую изображение {i}/{len(images)}: {filename}")
                
                description = await self.analyzer.analyze_image(image_data, filename)
                results.append((filename, description))
                images_with_analysis.append((filename, image_data, description))
                
                # Небольшая задержка для избежания rate limit
                await asyncio.sleep(0.5)
            
            # Создаем переименованный ZIP архив
            await processing_message.edit_text("📦 Создаю переименованный архив...")
            renamed_zip_data = await self.create_renamed_zip(images_with_analysis)
            
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
            
            # Отправляем переименованный ZIP архив
            await update.message.reply_text(
                "📤 *Переименованный архив готов к скачиванию!*\n\n"
                "📁 Файлы переименованы согласно результатам анализа\n"
                "📋 В архиве есть README с расшифровкой схемы именования",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Создаем имя для нового архива
            original_name = document.file_name.replace('.zip', '')
            new_archive_name = f"{original_name}_analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            # Отправляем архив
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=io.BytesIO(renamed_zip_data),
                filename=new_archive_name,
                caption="📂 Переименованные изображения с анализом"
            )
                
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

    async def create_renamed_zip(self, images_with_analysis: List[Tuple[str, bytes, str]]) -> bytes:
        """Создает ZIP архив с переименованными файлами"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            used_names = set()  # Для избежания дублирования имен
            
            for i, (original_filename, image_data, analysis_text) in enumerate(images_with_analysis):
                # Парсим результаты анализа
                analysis_results = self.analyzer.parse_analysis_results(analysis_text)
                
                # Создаем новое имя файла
                new_filename = self.analyzer.create_new_filename(original_filename, analysis_results)
                
                # Проверяем на дублирование и добавляем номер если нужно
                original_new_filename = new_filename
                counter = 1
                while new_filename in used_names:
                    name, ext = os.path.splitext(original_new_filename)
                    new_filename = f"{name}_{counter:03d}{ext}"
                    counter += 1
                
                used_names.add(new_filename)
                
                # Добавляем файл в архив
                zip_file.writestr(new_filename, image_data)
            
            # Добавляем файл с расшифровкой схемы именования
            readme_content = """Схема именования файлов:

R[0/1] - Реалистичное фото (1=да, 0=нет)
I[0/1] - Иллюстрация (1=да, 0=нет)  
[obj] - Основной объект (people/tech/product/другое)
[color] - Цвет фона (white/black/red/blue/и т.д.)
S[0/1] - Скидка/выгода (1=да, 0=нет)

Для GIF анимаций добавляется префикс "GIF-"
Анализ GIF проводится по ключевым кадрам (макс. 5 кадров)
Юридические дисклеймеры и предупреждения игнорируются

Примеры:
- R1-I0-people-blue-S0_photo123.jpg (обычное фото)
- GIF-R0-I1-tech-white-S1_banner.gif (GIF анимация)

Означает: 
- реалистичное фото/иллюстрация
- люди/техника как основной объект
- синий/белый фон
- нет/есть скидки/промо

Создано ботом анализа изображений
"""
            zip_file.writestr("README_naming_scheme.txt", readme_content)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

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