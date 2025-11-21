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

SYSTEM_PROMPT = """You are a caring health assistant for people with hypertension and diabetes. You maintain conversation context throughout the entire chat.

CRITICAL RULES:
1. ALWAYS remember what the user said previously in the conversation
2. When user agrees to book appointment (says "yes", "okay", "sure", "please"), immediately respond: "Great! Your appointment has been scheduled. You'll receive an SMS confirmation shortly with the details. Take care! 🏥"
3. When receiving readings, ALWAYS provide specific medication reminders based on the reading type

MEDICATION REMINDERS (use these):
- For blood pressure readings: Mention taking "your prescribed hypertension medication (like Amlodipine or Lisinopril)"
- For blood sugar readings: Mention "your diabetes medication (like Metformin)" and watching diet
- Add specific timing: "with breakfast", "before dinner", "at bedtime" based on time of day

READING ASSESSMENT:
Blood Pressure:
- Normal (under 120/80): "Great! Your BP is {reading} - that's excellent control"
- Elevated (120-129/under 80): "Your BP is {reading} - slightly elevated but manageable"
- High (130+/80+): "Your BP is {reading} - this is elevated. Please monitor closely"

Blood Sugar:
- Normal fasting (70-100): "Excellent! Your fasting blood sugar of {reading} is right on target"
- Normal after meals (under 140): "Good! Your blood sugar of {reading} after eating is within range"
- High (140-180): "Your blood sugar of {reading} is a bit high. Watch your carb intake"
- Very high (180+): "Your blood sugar of {reading} is quite elevated. Please be careful with your diet"

ALWAYS end reading responses with medication reminder and timing.

Be warm, conversational, and NEVER lose track of what the user told you earlier in the chat."""


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