"""Discord bot bridge — forwards student questions to the RAG engine."""
import discord
from discord.ext import commands


async def run_discord_bot(rag_engine, discord_cfg: dict):
    intents = discord.Intents.default()
    intents.message_content = True

    prefix = discord_cfg.get("prefix", "!ask")
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
            for s in sources[:5]:
                response += f"- [{s['title']}]({s['url']})\n"

        for i in range(0, len(response), 2000):
            await ctx.send(response[i : i + 2000])

    token = discord_cfg["token"]
    await bot.start(token)
