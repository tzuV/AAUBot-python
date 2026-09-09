"""Discord bot bridge — forwards student questions to the RAG engine."""
import discord
from discord.ext import commands


async def run_discord_bot(rag_engine, discord_cfg: dict, use_message_content=True):
    intents = discord.Intents.default()
    
    # Only enable message_content intent if explicitly requested and approved
    if use_message_content:
        intents.message_content = True
    else:
        # Without message_content intent, we can still see messages but not their content
        # Commands won't work, but the bot can respond to other events
        print("WARNING: message_content intent is disabled.")
        print("The bot will be online but cannot read message content.")
        print("To enable commands, set use_message_content=True and enable")
        print("'Message Content Intent' in Discord Developer Portal")
    
    intents.messages = True

    prefix = discord_cfg.get("prefix", "!")
    channel_id = discord_cfg.get("channel_id", "")

    bot = commands.Bot(
        command_prefix=commands.when_mentioned_or(prefix),
        intents=intents,
        help_command=None,
    )

    @bot.event
    async def on_ready():
        print(f"Discord bot logged in as {bot.user} (prefix: '{prefix}')")

    @bot.command(name="ask")
    async def ask(ctx, *, question: str):
        """Ask a Python question: !ask <question>"""
        if channel_id and str(ctx.channel.id) != str(channel_id):
            return

        async with ctx.typing():
            result = await rag_engine.ask(question)
            answer = result["answer"]
            sources = result["sources"]

        # Build response, respecting Discord's 2000-char message limit
        response = f"**Answer:**\n{answer}\n"
        if sources:
            response += "\n**Sources:**\n"
            for s in sources[:2]:
                response += f"- [{s['title']}]({s['url']})\n"

        for i in range(0, len(response), 2000):
            await ctx.send(response[i : i + 2000])

    token = discord_cfg["token"]
    await bot.start(token)
