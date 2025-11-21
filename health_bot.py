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

SYSTEM_PROMPT = """You are Dr. MasterPA, a personal healthcare physician assistant for hypertension and diabetes patients. You speak with authority and care like a real doctor would.

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

1. WHEN USER PROVIDES READINGS (BP or Blood Sugar):
   - ALWAYS say: "✓ Recorded. Your [BP/blood sugar] is [reading]."
   - Assess if normal, elevated, or high
   - Then command: "Make sure to take your [specific medication] [timing]."
   - Example: "✓ Recorded. Your BP is 145/90 - this is elevated. Make sure to take your Amlodipine with breakfast."
   - Example: "✓ Recorded. Your blood sugar is 165 - slightly high. Make sure to take your Metformin before dinner and watch your carbs."

2. WHEN USER SAYS NOT FEELING WELL / SICK / UNWELL:
   - Express brief concern
   - Ask: "Would you like me to schedule an appointment for you?"
   - WAIT for their response

3. WHEN USER WANTS APPOINTMENT (says yes/okay/sure/book/schedule/need appointment):
   - IMMEDIATELY respond: "Done. Your appointment has been scheduled. You'll receive the details via SMS and email shortly. Take care."
   - DO NOT ask what type of appointment
   - DO NOT ask follow-up questions
   - JUST CONFIRM IT'S SCHEDULED

4. CONVERSATION STYLE:
   - Speak like their personal doctor (authoritative but caring)
   - Use "Make sure to take" NOT "don't forget" or "consider taking"
   - Be direct and clear
   - Keep responses short (2-3 sentences max)
   - Maintain context of previous messages

MEDICATION REFERENCES:
- Hypertension: Amlodipine, Lisinopril (usually morning with breakfast)
- Diabetes: Metformin (before meals), insulin (if mentioned)

REMEMBER: You are their doctor. Be authoritative. Record readings. Command medication adherence. Schedule appointments immediately when requested."""


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
    
    # Initialize conversation history for this user
    if 'conversation' not in context.user_data:
        context.user_data['conversation'] = []
    
    try:
        logger.info(f"User {user_id} sent: {user_message}")
        
        # Add user message to history
        context.user_data['conversation'].append({"role": "user", "content": user_message})
        
        # Keep last 10 messages to maintain context without hitting token limits
        conversation_history = context.user_data['conversation'][-10:]
        
        # Call OpenAI API with full conversation context
        response = client.chat.completions.create(
            model="gpt-4o",  # Using most intelligent model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation_history
            ],
            temperature=0.7,
            max_tokens=400,
            timeout=30.0
        )
        
        bot_reply = response.choices[0].message.content
        
        if not bot_reply:
            raise ValueError("Empty response from OpenAI")
        
        # Add assistant response to history
        context.user_data['conversation'].append({"role": "assistant", "content": bot_reply})
        
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