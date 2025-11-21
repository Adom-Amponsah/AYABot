import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI, OpenAIError
from typing import Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client: Optional[OpenAI] = None

SYSTEM_PROMPT = """You are a friendly health assistant for people with hypertension and diabetes.

Your role:
- Help users log their daily readings (blood pressure like 150/90 or blood sugar levels)
- Provide brief, encouraging feedback on their readings
- If readings seem concerning (BP over 140/90 or blood sugar over 180), gently suggest they monitor closely
- If they mention feeling unwell, ask if they'd like to book an appointment with their doctor
- Be conversational, warm, and supportive - not clinical or robotic
- Keep responses short and natural (2-3 sentences max usually)

Guidelines for readings:
- Blood pressure: Normal is under 120/80, elevated is 120-129/80, high is 130+/80+
- Blood sugar: Normal fasting is 70-100, after meals under 140, over 180 is concerning

Be understanding that people may phrase things casually. Extract the numbers intelligently."""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    if not update.message:
        return
    
    try:
        welcome_message = (
            "Hi! I'm your health assistant. 👋\n\n"
            "You can share your blood pressure or blood sugar readings with me anytime, "
            "and I'll help you track how you're doing. Just chat naturally!\n\n"
            "Examples:\n"
            "• My BP is 130/85\n"
            "• Blood sugar 145 after lunch\n"
            "• I'm not feeling well today"
        )
        await update.message.reply_text(welcome_message)
        logger.info(f"User {update.effective_user.id} started the bot")
    except Exception as e:
        logger.error(f"Error in start command: {e}", exc_info=True)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages with OpenAI"""
    if not update.message or not update.message.text:
        return
    
    user_message = update.message.text
    user_id = update.effective_user.id
    
    try:
        logger.info(f"User {user_id} sent: {user_message}")
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300,
            timeout=30.0
        )
        
        bot_reply = response.choices[0].message.content
        
        if not bot_reply:
            raise ValueError("Empty response from OpenAI")
        
        await update.message.reply_text(bot_reply)
        logger.info(f"Successfully replied to user {user_id}")
        
    except OpenAIError as e:
        logger.error(f"OpenAI API error for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "I'm having trouble connecting right now. Please try again in a moment."
        )
    except Exception as e:
        logger.error(f"Unexpected error for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "Something went wrong. Please try again."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot"""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)


def validate_environment() -> tuple[str, str]:
    """Validate required environment variables"""
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return telegram_token, openai_key


def main() -> None:
    """Start the bot"""
    global client
    
    try:
        # Validate environment and initialize clients
        telegram_token, openai_key = validate_environment()
        client = OpenAI(api_key=openai_key)
        
        # Create application
        app = Application.builder().token(telegram_token).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_error_handler(error_handler)
        
        # Start bot
        logger.info("✓ Health Assistant Bot is running...")
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()